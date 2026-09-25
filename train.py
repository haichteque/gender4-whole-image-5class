"""Train the 5-class gender4 whole-image classifier.

Expects ImageFolder layout under --data:
  train/{real_male,real_female,anime_male,anime_female,other}/
  val/...same...

Example:
  python train.py --data data --epochs 20 --out checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import IMG_SIZE, LABELS, NUM_CLASSES, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train gender4 5-class classifier")
    p.add_argument("--data", type=Path, default=Path("data"), help="Root with train/ and val/")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Classifier head LR")
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/best.pt"))
    return p.parse_args()


def make_loaders(data_root: Path, batch_size: int, workers: int):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir():
        raise SystemExit(f"Missing train folder: {train_dir}")
    if not val_dir.is_dir():
        raise SystemExit(f"Missing val folder: {val_dir}")

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),  # [0,1] — ImageNet norm is inside the model
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    # Enforce fixed label order (ImageFolder sorts alphabetically otherwise).
    expected = {name: i for i, name in enumerate(LABELS)}
    for name in LABELS:
        if name not in train_ds.class_to_idx:
            raise SystemExit(
                f"Train folder missing class '{name}'. Found: {list(train_ds.class_to_idx)}"
            )
        if name not in val_ds.class_to_idx:
            raise SystemExit(
                f"Val folder missing class '{name}'. Found: {list(val_ds.class_to_idx)}"
            )

    def remap(ds: datasets.ImageFolder) -> None:
        # Remap targets so index matches LABELS order, not alphabetical.
        old = ds.class_to_idx
        new_samples = []
        for path, old_idx in ds.samples:
            cls_name = next(k for k, v in old.items() if v == old_idx)
            new_samples.append((path, expected[cls_name]))
        ds.samples = new_samples
        ds.targets = [t for _, t in new_samples]
        ds.class_to_idx = expected
        ds.classes = list(LABELS)

    remap(train_ds)
    remap(val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for c in range(NUM_CLASSES):
            mask = y == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (pred[mask] == c).sum().item()
    acc = correct / max(total, 1)
    per_class = {
        LABELS[c]: (per_class_correct[c] / per_class_total[c] if per_class_total[c] else 0.0)
        for c in range(NUM_CLASSES)
    }
    return {"acc": acc, "per_class": per_class, "n": total}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} labels={LABELS}")

    train_loader, val_loader, n_train, n_val = make_loaders(
        args.data, args.batch_size, args.workers
    )
    print(f"train={n_train} val={n_val}")

    model = build_model(pretrained=True).to(device)
    crit = nn.CrossEntropyLoss()
    head_params = list(model.base.classifier.parameters())
    back_params = list(model.base.features.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": back_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()

        train_loss = running / max(seen, 1)
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_acc={metrics['acc']*100:.2f}% "
            f"per_class={{{', '.join(f'{k}={v*100:.1f}%' for k, v in metrics['per_class'].items())}}}"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_acc,
                    "labels": LABELS,
                },
                args.out,
            )
            print(f"  saved {args.out} (val_acc={best_acc*100:.2f}%)")

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"best_val_acc": best_acc, "history": history, "labels": LABELS}, indent=2),
        encoding="utf-8",
    )
    print(f"done. best_val_acc={best_acc*100:.2f}% → {args.out}")


if __name__ == "__main__":
    main()
