# PAMCL Training on Kaggle - Quick Guide

## Step 1: Prepare the Code

1. **Zip the pamcl folder:**
   ```
   # On Windows (PowerShell):
   Compress-Archive -Path "D:\My_Files\Research\pamcl\*" -DestinationPath "D:\My_Files\Research\pamcl.zip"
   ```

2. **Upload to Kaggle:**
   - Go to kaggle.com → Datasets → New Dataset
   - Name it: `pamcl-code`
   - Upload `pamcl.zip`

## Step 2: Find Dataset Paths

Your ASVSpoof 2021 datasets on Kaggle will have paths like:
```
/kaggle/input/<your-dataset-name>/...
```

Common structures:
- `/kaggle/input/asvspoof2021-la/ASVspoof2021_LA_eval/flac/`
- `/kaggle/input/asvspoof2021-la-keys/keys/LA/CM/trial_metadata.txt`

## Step 3: Create Notebook

1. Go to Kaggle → Code → New Notebook
2. **Settings (right panel):**
   - Accelerator: **GPU T4 x2**
   - Language: Python
   - Persistence: Files
3. Add your datasets:
   - pamcl-code (your uploaded code)
   - ASVSpoof 2021 LA audio
   - ASVSpoof 2021 LA keys

## Step 4: Upload and Run

1. Upload `kaggle_train.ipynb` or copy cells manually
2. **Update paths in Section 3** to match your dataset structure
3. Run all cells

## Expected Training Time (2x T4)

| Epochs | Time |
|--------|------|
| 5 | ~8-10 hours |
| 20 | ~32-40 hours |
| 35 (research) | ~2-3 days |

## Tips

### Prevent Timeout:
- Kaggle notebooks timeout after 12 hours (free) or 9 hours idle
- Use "Save & Run All (Commit)" to run in background
- Save checkpoints every 5 epochs

### Resume Training:
```python
# In notebook, add before training:
checkpoint = torch.load('/kaggle/working/checkpoints/last_checkpoint.pt')
trainer.model.load_state_dict(checkpoint['model_state_dict'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.current_epoch = checkpoint['epoch']
```

### Download Results:
After training, checkpoints are in `/kaggle/working/checkpoints/`
- Download manually, or
- The notebook zips them to `pamcl_checkpoints.zip`

## Troubleshooting

### OOM Error:
Reduce batch_size in kaggle_config:
```python
'batch_size': 16,  # Instead of 32
```

### Import Errors:
Check that pamcl.zip extracted correctly:
```python
!ls /kaggle/working/pamcl/
```

### Dataset Not Found:
List available datasets:
```python
!find /kaggle/input -maxdepth 3 -type d
```
