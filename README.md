# ARC3Net

Reliability-guided crowd counting implementation based on a MambaVision backbone.

The current optimized implementation is centered on:

- `Module/mamba_reliability_v4.py`: reliability-guided two-stage model.
- `reliability_losses.py`: supervised and semi-supervised reliability losses.
- `train_full_supervised.py`: full-supervision baseline training entry.
- `train_sem.py`: original semi-supervised training entry kept for comparison.

Large model weights, checkpoints, datasets, IDE metadata, and cache files are intentionally excluded from this repository.

## Main Idea

ARC3Net keeps the reliability-guided training path:

1. Learn foreground and reliability cues from supervised density information.
2. Use reliability to control adaptive fusion and relation residuals.
3. Fall back to stable fusion in low-reliability regions.
4. Weight teacher-student consistency by teacher reliability in both spatial and global strength.

## Requirements

This project depends on PyTorch, TorchVision, MambaVision-related packages, and common data processing libraries. The Mamba SSM backend is CUDA/Linux-oriented.

```bash
pip install -r requirements.txt
```

## Full Supervision

```bash
python train_full_supervised.py \
  --data_root /data/LM/Dataset \
  --dataset_name SHA \
  --gpu_id 0
```

## Notes

- Put pretrained weights under `weights/` locally; they are ignored by Git.
- Dataset paths are configured in `dataset/loaddata.py`.
- The new model/loss files are not wired into the legacy training script automatically, so the old implementation remains available for controlled comparison.

