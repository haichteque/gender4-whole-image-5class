# Kaggle notebook outline

Copy these cells into a Kaggle Notebook with **GPU** on. Attach your ImageFolder
dataset and (optionally) this repo as a dataset / notebook input.

## Cell 1 — setup

```python
!pip install -q onnx onnxruntime tqdm
import os, sys
sys.path.insert(0, "/kaggle/input/gender4-whole-image-5class")  # adjust dataset name
os.chdir("/kaggle/working")
```

## Cell 2 — train

```python
!python /kaggle/input/gender4-whole-image-5class/train.py \
  --data /kaggle/input/YOUR_DATASET \
  --epochs 20 \
  --batch-size 64 \
  --out /kaggle/working/best.pt
```

## Cell 3 — eval (watch other→person FP rate)

```python
!python /kaggle/input/gender4-whole-image-5class/eval.py \
  --data /kaggle/input/YOUR_DATASET/val \
  --checkpoint /kaggle/working/best.pt
```

## Cell 4 — export ONNX

```python
!python /kaggle/input/gender4-whole-image-5class/export_onnx.py \
  --checkpoint /kaggle/working/best.pt \
  --out /kaggle/working/model.onnx
```

Download `/kaggle/working/model.onnx` when finished.
