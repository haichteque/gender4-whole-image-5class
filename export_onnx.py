"""Export a trained gender4 checkpoint to ONNX.

Contract:
  input  float32 [1, 3, 256, 256]  RGB CHW in [0, 1]
  output float32 [1, 5]            softmax probabilities (LABELS order)

Example:
  python export_onnx.py --checkpoint checkpoints/best.pt --out model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from model import IMG_SIZE, LABELS, NUM_CLASSES, SoftmaxWrapper, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export gender4 checkpoint to ONNX")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("model.onnx"))
    p.add_argument("--opset", type=int, default=13)
    return p.parse_args()


def export(checkpoint: Path, out: Path, opset: int) -> None:
    backbone = load_checkpoint(checkpoint)
    wrapped = SoftmaxWrapper(backbone).eval()

    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    # Collapse any external data into a single file.
    model = onnx.load(str(out), load_external_data=True)
    onnx.save(model, str(out))

    # Sanity: shapes + softmax-ish sum
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    oup = sess.get_outputs()[0]
    assert inp.name == "input", inp.name
    assert oup.name == "output", oup.name
    assert list(inp.shape) == [1, 3, IMG_SIZE, IMG_SIZE], inp.shape
    assert list(oup.shape) == [1, NUM_CLASSES], oup.shape

    x = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y = sess.run(None, {"input": x})[0]
    assert y.shape == (1, NUM_CLASSES), y.shape
    s = float(y.sum())
    assert 0.99 <= s <= 1.01, f"softmax sum={s}"

    mb = out.stat().st_size / 1e6
    print(f"exported {out} ({mb:.2f} MB)")
    print(f"  input={inp.name} {list(inp.shape)}")
    print(f"  output={oup.name} {list(oup.shape)} labels={LABELS}")
    print(f"  sample softmax sum={s:.4f}")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    export(args.checkpoint, args.out, args.opset)


if __name__ == "__main__":
    main()
