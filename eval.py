"""Evaluate a gender4 checkpoint on an ImageFolder validation set.

Reports overall accuracy, per-class recall, and confusion matrix.
Also reports the share of `other` images predicted as a person class
(false-positive rate proxy for downstream actions).

Example:
  python eval.py --data data/val --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMG_SIZE, LABELS, NUM_CLASSES, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval gender4 checkpoint")
    p.add_argument("--data", type=Path, required=True, help="ImageFolder root (val/)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def load_folder(data: Path, batch_size: int, workers: int) -> DataLoader:
    if not data.is_dir():
        raise SystemExit(f"data folder not found: {data}")
    tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(str(data), transform=tf)
    expected = {name: i for i, name in enumerate(LABELS)}
    for name in LABELS:
        if name not in ds.class_to_idx:
            raise SystemExit(
                f"Missing class folder '{name}'. Found: {list(ds.class_to_idx)}"
            )
    new_samples = []
    for path, old_idx in ds.samples:
        cls_name = next(k for k, v in ds.class_to_idx.items() if v == old_idx)
        new_samples.append((path, expected[cls_name]))
    ds.samples = new_samples
    ds.targets = [t for _, t in new_samples]
    ds.class_to_idx = expected
    ds.classes = list(LABELS)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers
    ), len(ds)


@torch.no_grad()
def run_eval(model, loader, device) -> dict:
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for t, p in zip(y.tolist(), pred.tolist()):
            conf[t, p] += 1

    conf_list = conf.tolist()
    totals = conf.sum(dim=1)
    correct = conf.diag()
    per_class = {}
    for i, name in enumerate(LABELS):
        n = int(totals[i].item())
        per_class[name] = {
            "n": n,
            "recall": float(correct[i].item() / n) if n else 0.0,
        }
    total = int(conf.sum().item())
    acc = float(correct.sum().item() / total) if total else 0.0

    other_i = LABELS.index("other")
    other_n = int(totals[other_i].item())
    other_as_person = int(conf[other_i].sum().item() - conf[other_i, other_i].item())
    other_fp_rate = (other_as_person / other_n) if other_n else 0.0

    return {
        "acc": acc,
        "n": total,
        "per_class": per_class,
        "confusion": {LABELS[i]: conf_list[i] for i in range(NUM_CLASSES)},
        "other_predicted_as_person_rate": other_fp_rate,
        "labels": LABELS,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, map_location=device).to(device)
    loader, n = load_folder(args.data, args.batch_size, args.workers)
    print(f"eval n={n} device={device} checkpoint={args.checkpoint}")

    metrics = run_eval(model, loader, device)
    print(f"accuracy={metrics['acc']*100:.2f}%")
    for name, info in metrics["per_class"].items():
        print(f"  {name}: recall={info['recall']*100:.1f}% (n={info['n']})")
    print(
        f"other→person FP rate={metrics['other_predicted_as_person_rate']*100:.2f}% "
        "(gate: aim <1–2% before deploying actions)"
    )
    print("confusion rows=true cols=pred order:", LABELS)
    for name, row in metrics["confusion"].items():
        print(f"  {name}: {row}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
