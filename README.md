# PAMCL — Phoneme-Aware Multi-Layer Contrastive Learning for Audio Deepfake Detection

A deepfake speech detection model that combines SSL feature extraction (XLS-R 300M), phoneme-aware contrastive learning, and sensitive layer selection (SLS) for robust, generalizable detection.

## Architecture

- **SSL Backbone**: XLS-R 300M (frozen) — extracts features from all transformer layers
- **Phoneme Encoder**: CTC-based phoneme extraction using wav2vec2-base-960h
- **SLS Module**: Learns per-layer importance weights via sigmoid gating
- **Contrastive Head**: Phoneme-aware contrastive loss with prototype speakers
- **Classification Head**: Attention-pooled binary classifier (real vs fake)

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~3 GB disk for XLS-R model weights (auto-downloaded on first run)

### GPU Memory Requirements

| GPU | batch_size | gradient_accumulation_steps |
|-----|-----------|---------------------------|
| H100 80GB | 256 | 1 |
| A100 40GB | 128 | 2 |
| T4 16GB | 32 | 8 |
| RTX 3050 4GB | 8 | 32 |

## Setup

### 1. Clone and Create Virtual Environment

```bash
git clone <repo-url>
cd pamcl
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If you need a specific CUDA version for PyTorch:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
> pip install -r requirements.txt
> ```

### 3. Prepare Dataset

Download the **ASVSpoof 2021 LA** dataset and place it so the directory structure looks like:

```
Datasets/
├── ASVSpoof21/
│   ├── ASVspoof2021_LA_eval/
│   │   └── ASVspoof2021_LA_eval/
│   │       └── flac/         # .flac audio files
│   ├── ASVspoof2021_DF_eval_part00/
│   │   └── ASVspoof2021_DF_eval/
│   │       └── flac/
│   ├── LA-keys-full/
│   │   └── keys/LA/CM/
│   │       └── trial_metadata.txt
│   └── DF-keys-full/
│       └── keys/DF/CM/
│           └── trial_metadata.txt
└── MLAAD/                    # Optional: MLAAD v5 for evaluation
    └── Mlaad_v5/mlaad_v5/fake/
```

### 4. Update Config Paths

Edit `configs/default.yaml` and update the `data:` section with your dataset paths:

```yaml
data:
  train_dir: "/your/path/to/Datasets/ASVSpoof21/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac"
  train_protocol: "/your/path/to/Datasets/ASVSpoof21/LA-keys-full/keys/LA/CM/trial_metadata.txt"
  eval_dir: "/your/path/to/Datasets/ASVSpoof21/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac"
  eval_protocol: "/your/path/to/Datasets/ASVSpoof21/DF-keys-full/keys/DF/CM/trial_metadata.txt"
```

### 5. Run Pre-flight Test

Verify everything works before starting training:

```bash
python test_before_h100.py
```

All 12 tests should pass with ✅.

## Training

```bash
python train.py
```

Training uses a **2-phase** approach:
1. **Phase 1** (3 epochs): Phoneme contrastive pre-training
2. **Phase 2** (5 epochs): Multi-task fine-tuning with classification + contrastive losses

The train/val split is automatically created from LA eval (50,000 train + 7,000 val, stratified).

### Adjust for Your GPU

Edit `configs/default.yaml`:

```yaml
training:
  batch_size: 256              # Lower for smaller GPUs
  gradient_accumulation_steps: 1  # Increase to maintain effective batch size
```

### Resume from Checkpoint

Checkpoints are saved to `checkpoints/` every 5 epochs. Training auto-resumes if a checkpoint is found.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

## Project Structure

```
pamcl/
├── configs/          # YAML configuration files
├── data/             # Dataset loaders and augmentation
├── models/           # PAMCL model architecture
├── training/         # Trainer with multi-phase logic
├── utils/            # Logging, config, and helper utilities
├── train.py          # Main training entry point
├── evaluate.py       # Evaluation script
└── test_before_h100.py  # Pre-flight verification test
```

## H100 Optimizations

The following optimizations are automatically enabled when a CUDA GPU is detected:

- **TF32 matmul**: `torch.backends.cuda.matmul.allow_tf32 = True`
- **TF32 cuDNN**: `torch.backends.cudnn.allow_tf32 = True`
- **cuDNN benchmark**: `torch.backends.cudnn.benchmark = True`
- **Mixed precision**: FP16 via `torch.amp.autocast` + `GradScaler`
- **DataParallel**: Auto-enabled when multiple GPUs are detected
