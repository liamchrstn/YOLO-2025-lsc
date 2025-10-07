# YOLOv11 Mannequin Detection

YOLOv11 model for detecting mannequins in two classes: laying and standing. Optimized for NVIDIA Jetson deployment.

## Stack
- YOLOv11 (Ultralytics)
- PyTorch
- TensorRT optimization
- Python 3.10+

## Structure
The project separates source data from processed dataset for reproducible data preparation.

```
YOLO-2025-lsc/
├── source_data/original_labels/  # Original 4-class labels
├── dataset/
│   ├── images/                   # Training images
│   └── labels/                   # Generated 2-class labels
├── runs/                         # Training outputs
├── prepare_dataset.py            # Data preprocessing
├── train.py                      # Training script
└── mannequin_dataset.yaml        # Dataset config
```

## Setup

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Install CUDA-enabled PyTorch (check nvidia-smi for CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data Preparation

The original dataset contains 4 classes: `bucket`, `mannequins laying`, `mannequins standing`, `people`. The preparation script filters and remaps to 2 classes for training.

**Class Mapping:**
- `mannequins laying` (class 1) → class 0
- `mannequins standing` (class 2) → class 1

```bash
# Place original labels in source_data/original_labels/
# Place images in dataset/images/
python prepare_dataset.py
```

Cleaned labels are generated in `dataset/labels/` without modifying source data.

## Training

Two training configurations available:

```bash
# Quick testing (50 epochs, 416px, batch 16)
python train.py --version toy

# Production training (100 epochs, 640px, batch 8, augmentations)
python train.py --version real
```

Results saved in `runs/` directory. Models automatically exported to ONNX and TensorRT formats.

## Scripts

- `prepare_dataset.py` - Convert 4-class labels to 2-class labels
- `train.py` - Train YOLOv11 model with automatic export
- `mannequin_dataset.yaml` - Dataset configuration
