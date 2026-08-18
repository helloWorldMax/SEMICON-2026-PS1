# AI-Based Restoration of Degraded Images

## SEMICON India Hackathon 2026 — PS1

This project develops a CNN-based image restoration system for recovering clean, high-resolution grayscale images from degraded low-resolution inputs.

## Problem

The input images contain degradation such as noise and reduced spatial resolution. The goal is to reconstruct a clean 256×256 image from a degraded 128×128 input.

## Dataset

The provided dataset contains paired training data:

- `NoisyLR`: 128×128 degraded images
- `GT`: 256×256 ground-truth images
- Image format: `.npy`
- Data type: `float32`

## Model

The project uses a PyTorch CNN with residual blocks for image restoration.

Pipeline:

NoisyLR → Upsampling → CNN Restoration → 256×256 Restored Image

## Files

- `dataset.py` — Dataset loading and preprocessing
- `model.py` — CNN restoration model
- `train.py` — Model training
- `test.py` — Test image restoration
- `requirements.txt` — Python dependencies

## Technologies

- Python
- PyTorch
- NumPy
- CNN
- Residual Learning

## Training

The model is trained using paired degraded and ground-truth images with L1 reconstruction loss and the Adam optimizer.

## Evaluation

The final solution will be evaluated using image restoration quality metrics such as:

- PSNR
- SSIM
- LPIPS
- Inference time

## Team

SEMICON India Hackathon 2026
