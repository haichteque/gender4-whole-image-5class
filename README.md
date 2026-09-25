# gender4-whole-image-5class

Train / eval / export scripts for a **MobileNetV2** whole-image classifier with five classes:

| Index | Label |
|------:|-------|
| 0 | `real_male` |
| 1 | `real_female` |
| 2 | `anime_male` |
| 3 | `anime_female` |
| 4 | `other` |

Built for an application that takes downstream actions from gender (and real vs anime) predictions. The `other` class covers non-person content (objects, animals, scenery) so those images are not treated as people.

**This repository is code only.** Released ONNX weights (and labels / metrics) live on Hugging Face:

<!-- HF_MODEL_URL -->
**Weights:** https://huggingface.co/YOUR_HF_USERNAME/gender4-whole-image-5class

Replace `YOUR_HF_USERNAME` (or the whole URL) after you publish the model.

## Scripts

| File | Role |
|------|------|
| `model.py` | Network definition + checkpoint load |
| `train.py` | ImageFolder training |
| `eval.py` | Accuracy, per-class recall, confusion, `other`→person FP rate |
| `export_onnx.py` | Export `best.pt` → `model.onnx` |
| `labels.json` | Canonical label order |
| `kaggle_notebook.md` | Example Kaggle GPU cells |

## Data layout

```text
data/
  train/{real_male,real_female,anime_male,anime_female,other}/
  val/...same folders...
```

## Kaggle workflow (GPU)

```bash
pip install -q onnx onnxruntime tqdm
python train.py --data /kaggle/input/YOUR_DATASET --epochs 20 --out /kaggle/working/best.pt
python eval.py --data /kaggle/input/YOUR_DATASET/val --checkpoint /kaggle/working/best.pt
python export_onnx.py --checkpoint /kaggle/working/best.pt --out /kaggle/working/model.onnx
```

Before deploying actions, check `eval.py`: keep `other`→person false positives low (aim under ~1–2%).

## ONNX contract

- **Input** `input`: `float32` `[1, 3, 256, 256]`, RGB CHW in `[0, 1]` (`/255`). ImageNet mean/std is inside the graph.
- **Output** `output`: `float32` `[1, 5]`, softmax probabilities in `labels.json` order.

## License

MIT
