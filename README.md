# YOLOv11 Mannequin Detection for Drone Deployment

This project trains a YOLOv11 object detection model to identify two classes of mannequins: "laying" and "standing". The workflow is designed to be robust, repeatable, and optimized for eventual deployment on an NVIDIA Jetson device mounted on a drone.

It includes a non-destructive data preparation pipeline that cleans and remaps a multi-class source dataset into a clean, two-class dataset suitable for training.

## Table of Contents
1.  [Core Technologies](#core-technologies)
2.  [Project Structure](#project-structure)
3.  [Setup and Installation](#setup-and-installation)
4.  [Data Preparation Workflow](#data-preparation-workflow)
5.  [Model Training](#model-training)
6.  [Model Export for Deployment](#model-export-for-deployment)
7.  [Key Scripts Overview](#key-scripts-overview)
8.  [Troubleshooting](#troubleshooting)

## Core Technologies
*   **Model:** YOLOv11 (via Ultralytics)
*   **Framework:** PyTorch
*   **Deployment Target:** NVIDIA Jetson (with TensorRT optimization)
*   **Scripting:** Python 3

## Project Structure
The project is organized to separate raw source data from the clean, processed dataset used for training. This ensures data integrity and makes the preparation process repeatable.

```
YOLO-2025-lsc/
├── source_data/
│   └── original_labels/      <-- Your original, untouched 4-class labels go here.
│       ├── train/
│       └── val/
│
├── dataset/
│   ├── images/               <-- Your training and validation images.
│   │   ├── train/
│   │   └── val/
│   └── labels/               <-- This is FILLED BY THE SCRIPT with clean, 2-class labels.
│       ├── train/
│       └── val/
│
├── runs/                       <-- Training results (models, logs, charts) are saved here.
│   ├── toy/
│   └── real/
│
├── prepare_dataset.py        <-- SCRIPT: Cleans and remaps source labels into the dataset folder.
├── train.py                  <-- SCRIPT: Trains the YOLOv11 model.
├── mannequin_dataset.yaml      <-- CONFIG: Tells the trainer where to find the clean data.
├── requirements.txt          <-- Project dependencies.
├── yolo11n.pt                <-- Pre-trained model weights.
└── yolo11s.pt                <-- Pre-trained model weights.```

## Setup and Installation

### 1. Prerequisites
*   Python 3.10+
*   NVIDIA GPU with CUDA drivers installed (for GPU training)

### 2. Environment Setup
It is highly recommended to use a Python virtual environment to manage dependencies.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# On Windows:
.\venv\Scripts\Activate
# On macOS/Linux:
# source venv/bin/activate
```

### 3. Install Dependencies
A `requirements.txt` file should be created to manage dependencies.

**`requirements.txt`:**```
ultralytics
torch
torchvision
torchaudio
```

Install these using pip:
```bash
pip install -r requirements.txt
```

### 4. GPU-Accelerated PyTorch (Crucial for Performance)
If training on an NVIDIA GPU (like a 1070 Ti), you must install the CUDA-enabled version of PyTorch. The standard `pip install torch` often installs the CPU-only version.

1.  **Check your CUDA Driver Version:**
    ```bash
    nvidia-smi
    ```
    Look at the "CUDA Version" in the top right.

2.  **Install the Correct PyTorch Build:** Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct command. For a system with CUDA 11.8, the command is:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    This step is critical for avoiding `torch.cuda.is_available(): False` errors.

## Data Preparation Workflow

### The Challenge
The original dataset was annotated with four classes: `0: bucket`, `1: mannequins laying`, `2: mannequins standing`, and `3: people`. Our goal is to train a model that only detects the two mannequin classes. Simply ignoring the extra classes during training is not ideal, as the label files themselves are considered invalid by the YOLO trainer if they contain class IDs outside the specified range.

### The Solution: A Non-Destructive Pipeline
We use the `prepare_dataset.py` script to create a clean, perfectly formatted dataset for our two-class problem without destroying the original four-class labels.

This script performs two key actions:
1.  **Filters:** It reads the original label files and discards any annotations that are not for `mannequins laying` (class 1) or `mannequins standing` (class 2).
2.  **Remaps:** It shifts the class IDs to fit our new two-class scheme:
    *   `mannequins laying` (old class `1`) becomes the **new class `0`**.
    *   `mannequins standing` (old class `2`) becomes the **new class `1`**.

### How to Prepare the Data
1.  **Place Original Data:** Copy your original, untouched label folders into `source_data/original_labels/`.
2.  **Place Images:** Ensure your corresponding images are in `dataset/images/`.
3.  **Run the Script:** Execute the preparation script from your terminal.
    ```bash
    python prepare_dataset.py
    ```
4.  **Verify:** After the script runs, the `dataset/labels/` directory will be populated with the new, cleaned, and remapped label files. Your original data in `source_data` remains untouched.

## Model Training

The `train.py` script is used to start the training process. It supports two distinct modes, selectable via a command-line argument.

### Training Modes

1.  **`toy` version (Default):**
    *   **Purpose:** Quick testing, debugging the pipeline, and rapid prototyping.
    *   **Model:** `yolo11n.pt` (nano), the smallest and fastest model.
    *   **Settings:** Lower epoch count (50), smaller image size (416x416).
    *   **Command:**
        ```bash
        python train.py --version toy
        ```
        *(Or simply `python train.py`)*

2.  **`real` version:**
    *   **Purpose:** Training a production-ready model for deployment.
    *   **Model:** `yolo11s.pt` (small), offering a better balance of speed and accuracy.
    *   **Settings:** Higher epoch count (100), larger image size (640x640), and extensive data augmentation to improve model robustness.
    *   **Command:**
        ```bash
        python train.py --version real
        ```

### Training Output
The results of each training run are saved in the `runs/` directory. Inside, you will find logs, performance charts, and the trained model weights. The most important file is **`best.pt`**, which represents the model with the best validation performance.

## Model Export for Deployment
The `train.py` script automatically handles exporting the final `best.pt` model to formats optimized for deployment:

*   **ONNX (`.onnx`):** A standard, widely-supported format for deep learning models.
*   **TensorRT (`.engine`):** An optimized format specifically for NVIDIA GPUs, providing significant speed boosts on devices like the Jetson.

These exported files will be located in the same directory as your `best.pt` model weights.

## Key Scripts Overview

*   **`prepare_dataset.py`**: Your dedicated data preparation tool. **Run this first.** It reads from `source_data`, applies the filter-and-remap logic, and writes the clean dataset to `dataset/labels`.
*   **`train.py`**: The main training script. It uses the clean data prepared by the previous script to train the YOLOv11 model. It is controlled via the `--version` command-line argument.
*   **`mannequin_dataset.yaml`**: The dataset configuration file. It tells the YOLO trainer where to find the clean images and labels, and defines the final class names (`mannequin_laying`, `mannequin_standing`).

## Troubleshooting

### `ValueError: Invalid CUDA 'device=auto' requested` or `torch.cuda.is_available(): False`
This is the most common issue and means your PyTorch installation cannot detect your NVIDIA GPU.

*   **Cause:** You likely installed the CPU-only version of PyTorch.
*   **Solution:** Follow the steps in the [GPU-Accelerated PyTorch](#4-gpu-accelerated-pytorch-crucial-for-performance) section to uninstall the current version and reinstall the correct, CUDA-enabled build.