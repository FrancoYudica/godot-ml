# MLP - Color Transform

Trains a tiny pixel-wise MLP (3→8→3) to approximate a color transformation, then exports the result as an ONNX model ready to load into the godot-ml engine.

## Setup

```bash
cd training/MLP
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Usage

```bash
# Train the default transform (invert) and export
python train.py

# Pick a different transform
python train.py --transform sepia

# Save a side-by-side preview image alongside the ONNX file
python train.py --transform warm_tone --preview assets/city.png
```

Available transforms: `warm_tone`, `grayscale`, `sepia`, `reinhard`, `invert`, `gamma`.

## Options

| Flag            | Default          | Description                                                          |
| --------------- | ---------------- | -------------------------------------------------------------------- |
| `--transform`   | `invert`         | Color transformation to learn                                        |
| `--epochs`      | `100`            | Training epochs                                                      |
| `--num-samples` | `100000`         | Random training pixels                                               |
| `--batch-size`  | `1024`           | Batch size                                                           |
| `--lr`          | `0.001`          | Learning rate                                                        |
| `--output`      | `color_mlp.onnx` | Output ONNX file path                                                |
| `--preview`     | —                | Image path; saves a `*_preview.png` comparison next to the ONNX file |

Generated models are in `models`
