# WBM Framework (CVAE + Hybrid Teacher + Distillation)

This repo provides a small framework to:
- train a **Conv-VAE (CVAE)** for denoising,
- train a **hybrid teacher** using **[original, denoised]** concatenated inputs (2-channel),
- train a **student** via **knowledge distillation** from the teacher,
- and **visualize** *Original / Radon / Generated* samples.

---

## Screenshots

> Put the two images below into an `assets/` folder so they render in this README.

### 1) Main GUI
![Main GUI](assets/gui_main.png)

### 2) Visualization Example
![Visualization Example](assets/viz_example.png)

**Expected filenames**
- `assets/gui_main.png`
- `assets/viz_example.png`

---

## Project Layout

```text
wf_cls/
├── launch.py                      # GUI launcher (project root)
├── code/
│   ├── train_cvae.py              # CVAE training (difficulty-aware)
│   ├── visualize_data.py          # Original/Radon/Generated visualization (difficulty-aware CVAE path)
│   ├── teacher_selection.py       # Base (1ch) teacher candidate training/eval
│   ├── train_teacher.py           # Hybrid (2ch) teacher training/eval (CVAE denoise + concat)
│   └── distillation.py            # Teacher → Student distillation (seed-suffix teacher rule)
├── data/
│   └── {difficulty}/{task}/{data_type}/...    # .npy data
├── weight/
│   ├── cvae/{difficulty}/best.h5              # CVAE weights (weights-only file)
│   ├── pretrained/levit384_imagenet.h5        # LeViT pretrained (optional)
│   └── {model}/{task}/{data_type}/...         # teacher/student weights
└── result/
    └── {task}/{data_type}/{model}/...         # metric csv logs
```

---

## Environment Setup

### 1) Create & activate env (example)
```bash
conda create -n wf_cls python=3.9 -y
conda activate wf_cls
pip install -r requirements.txt
```

### 2) (Optional) LeViT pretrained weights
If you want LeViT to load pretrained weights automatically, place the file here:
```text
weight/pretrained/levit384_imagenet.h5
```

---

## Data Convention (Important)

All scripts expect this naming scheme:

```text
data/{difficulty}/{task}/{data_type}/{task}_{split}_data.npy
data/{difficulty}/{task}/label/{task}_{split}_label.npy
```

- `difficulty`: e.g. `data_easy`, `data_hard`, `data_extreme`
- `task`: e.g. `mixed`, `multi`, `single`
- `data_type`: e.g. `radon`, `orig`
- `split`: `train`, `val`, `test`

---

## Run with the GUI (Recommended)

From the project root:
```bash
python launch.py
```

The GUI runs scripts located in `code/`.

---

## Run from CLI (Direct Execution)

> Run commands from the **project root** (`wf_cls/`) so relative paths resolve correctly.

### 1) Train CVAE (`code/train_cvae.py`)
Default output (weights-only):
```text
weight/cvae/{difficulty}/best.h5
```

Example:
```bash
python code/train_cvae.py \
  --difficulty data_hard \
  --epochs 300 \
  --batch 16 \
  --seed 1
```

---

### 2) Visualize (`code/visualize_data.py`)
Shows **3 rows**: `Original / Radon / Generated`

CVAE weights path resolution:
- If `--cvae_path` is provided → use it
- Else → `weight/cvae/{difficulty}/best.h5`

Example:
```bash
python code/visualize_data.py \
  --difficulty data_hard \
  --task mixed \
  --split test \
  --data_type radon \
  --orig_data_type orig \
  --seed 0 \
  --show
```

Save to PNG (optional):
```bash
python code/visualize_data.py \
  --difficulty data_hard \
  --task mixed \
  --split test \
  --save_path assets/viz_example.png
```

---

### 3) Train/Eval teacher candidates (base 1-channel) (`code/teacher_selection.py`)
Weights output:
```text
weight/{model}/{task}/{data_type}/best_{seed}.weights.h5
```

Example:
```bash
python code/teacher_selection.py \
  --mode train \
  --difficulty data_extreme \
  --task mixed \
  --data_type radon \
  --model levit \
  --seed 1
```

---

### 4) Train/Eval hybrid teacher (2-channel) (`code/train_teacher.py`)
Hybrid input is built as:
```text
hybrid_x = concat([original(1ch), denoised(1ch)], axis=-1)  # => 2ch
```

Teacher weights output (default):
```text
weight/{model}/{task}/{data_type}/best_{seed}_hybrid.h5
```

Example:
```bash
python code/train_teacher.py \
  --mode train \
  --difficulty data_extreme \
  --task mixed \
  --data_type radon \
  --model fastervit \
  --seed 1
```

---

### 5) Distillation (teacher → student) (`code/distillation.py`)
**Teacher file naming rule (important):**
- There is **no teacher_suffix**
- The **teacher seed** is used as the suffix

Default teacher path (auto):
```text
weight/{teacher_model}/{task}/{data_type}/best_{teacher_seed}_hybrid.h5
```

Student output:
```text
weight/{model}/{task}/{data_type}/best_{seed}_distill.h5
```

Example:
```bash
python code/distillation.py \
  --mode train \
  --difficulty data_extreme \
  --task mixed \
  --data_type radon \
  --teacher_model fastervit \
  --teacher_seed 1 \
  --model levit \
  --seed 1
```

Override teacher path manually:
```bash
python code/distillation.py \
  --difficulty data_extreme \
  --task mixed \
  --data_type radon \
  --teacher_path weight/fastervit/mixed/radon/best_1_hybrid.h5 \
  --model levit \
  --seed 1
```

---

## Outputs & Logs

### Weights
- CVAE: `weight/cvae/{difficulty}/best.h5`
- Base teacher candidates: `weight/{model}/{task}/{data_type}/best_{seed}.weights.h5`
- Hybrid teacher: `weight/{model}/{task}/{data_type}/best_{seed}_hybrid.h5`
- Distilled student: `weight/{model}/{task}/{data_type}/best_{seed}_distill.h5`

### Metrics CSV
Saved to:
```text
result/{task}/{data_type}/{model}/
```

---

## Troubleshooting

### `ModuleNotFoundError: keras_cv_attention_models`
Run from the project root:
```bash
python code/train_teacher.py ...
```
Or use the GUI (`python launch.py`).

### `FileNotFoundError: pretrained/levit384_imagenet.h5`
Place the file here:
```text
weight/pretrained/levit384_imagenet.h5
```

### CVAE weights path mismatch
Either:
- place weights at `weight/cvae/{difficulty}/best.h5`, or
- pass `--cvae_path /full/or/relative/path/to/best.h5`

---

## How to attach your screenshots
1) Create a folder: `assets/`
2) Save:
   - GUI screenshot → `assets/gui_main.png`
   - Visualization screenshot → `assets/viz_example.png`
3) Commit both images along with this README.