# YOLOv11 Mannequin Detection

A YOLOv11 project to detect mannequins, optimized for NVIDIA Jetson deployment. This project includes an optional self-supervised pre-training step using DINO to improve model accuracy.
## Project Structure

```
YOLO-2025-lsc/
├── dataset/
│   ├── images/train/       # Training images (unlabeled and labeled)
│   └── labels/train/       # Processed 2-class labels
├── source_data/            # Original raw data (e.g., 4-class labels)
├── runs/                   # All training and pre-training outputs
│
├── DINO-train.py           # Self-supervised pre-training script
├── prepare_dataset.py      # Prepares labels for supervised training
├── train.py                # Supervised fine-tuning script
│
├── mannequin_dataset.yaml  # Dataset configuration
└── requirements.txt        # Project dependencies
```

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Ensure PyTorch is installed with the correct CUDA version
    # Example for CUDA 11.8:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Workflow

The training process is two-phased: optional self-supervised pre-training followed by required supervised fine-tuning.

### 1. Data Preparation

The `prepare_dataset.py` script filters and remaps the original multi-class labels into the two target classes for detection.

**Class Mapping:**
- `mannequins laying` (class 1) → `mannequin_laying` (class 0)
- `mannequins standing` (class 2) → `mannequin_standing` (class 1)

**Execution:**
Place original labels in `source_data/original_labels/` and all corresponding images in `dataset/images/train/`.
```bash
python prepare_dataset.py
```
This generates cleaned labels in `dataset/labels/train/`.

### 2. Phase 1: Self-Supervised Pre-training (Optional)

This step uses DINO to teach the model's backbone to recognize visual features specific to your dataset, using only the raw images.

**Execution:**
```bash
# Quick pre-training run on a YOLOv11n backbone
python DINO-train.py --version toy

# Longer, more thorough pre-training on a YOLOv11s backbone
python DINO-train.py --version real
```
This produces a custom pre-trained weights file, e.g., `runs/pretrain_dino_toy/exported_models/exported_last.pt`.

### 3. Phase 2: Supervised Fine-tuning

This step trains the model to detect and classify mannequins using the prepared labels. You can start from default COCO weights or your custom DINO pre-trained weights.

**Option A: Fine-tune using custom DINO weights (Recommended)**
Provide the weights generated in Phase 1 to the training script. This will yield the best results.

```bash
python train.py --version toy --weights runs/pretrain_dino_toy/exported_models/exported_last.pt
```

**Option B: Train from default COCO weights**
If you skip Phase 1, you can train from the standard public weights.

```bash
# Quick 'toy' training run
python train.py --version toy

# Full 'real' training run with augmentations
python train.py --version real
```

All training runs automatically save results to the `runs/` directory and export the best model to ONNX and TensorRT formats.

## Jetson Deployment

1.  Copy the `best.onnx` file (found in a subdirectory of `runs/`) to your Jetson.
2.  On the Jetson, generate the optimized TensorRT engine:
    ```bash
    # Ensure ultralytics is installed: pip install ultralytics
    yolo export model=best.onnx format=engine half=True
    ```
3.  Load the `.engine` file in your application for maximum performance:
    ```python
    from ultralytics import YOLO
    
    model = YOLO('best.engine')
    ```

## Hyperparameter Tuning

For information on augmenting training parameters, refer to the official Ultralytics documentation:
[Augmentation Settings and Hyperparameters](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)