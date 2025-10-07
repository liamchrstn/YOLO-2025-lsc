"""
train.py: Supervised training script for YOLOv11 Mannequin Detection.
Can be initialized with standard weights or custom DINO pre-trained weights.
"""

import argparse
import torch
from ultralytics import YOLO
from pathlib import Path

# Configuration for 'toy' and 'real' fine-tuning runs
CONFIG = {
    'toy': {
        'default_model': 'yolo11n.pt', 
        'params': {
            'epochs': 50,
            'imgsz': 416,
            'batch': 16,
            'project': 'runs/toy',
            'name': 'mannequin_detection_toy',
            'patience': 15,
            'amp': True,
            'cache': True,
        }
    },
    'real': {
        'default_model': 'yolo11s.pt',
        'params': {
            'epochs': 100,
            'imgsz': 640,
            'batch': 8,
            'project': 'runs/real',
            'name': 'mannequin_detection_real',
            'patience': 25,
            'amp': True,
            'cache': True,
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
            'degrees': 10.0, 'translate': 0.1, 'scale': 0.5, 'shear': 2.0,
            'perspective': 0.0, 'flipud': 0.5, 'fliplr': 0.5,
            'mosaic': 1.0, 'mixup': 0.1,
        }
    }
}

def train(version='toy', data_path='mannequin_dataset.yaml', weights_path=None):
    """Train and export a YOLOv11 model."""
    if version not in CONFIG:
        print(f"Error: Invalid version '{version}'. Choose from {list(CONFIG.keys())}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting Supervised Fine-tuning ('{version}' version) on device: {device} ---")
    
    config = CONFIG[version]

    if weights_path:
        # Use custom pre-trained weights
        initial_model_path = Path(weights_path)
        print(f"Initializing model with custom pre-trained weights: {initial_model_path}")
        
        if not initial_model_path.exists():
            print(f"Error: Weights file not found at '{initial_model_path}'")
            return
    else:
        # Use default COCO-pre-trained weights from the root folder
        initial_model_path = config['default_model']
        print(f"Initializing model with default COCO weights: {initial_model_path}")

    # This simple constructor works when the library can find the necessary blueprint files.
    model = YOLO(initial_model_path)

    # Start the fine-tuning process
    print(f"Starting training with '{version}' parameters...")
    model.train(
        data=data_path,
        device=device,
        **config['params']
    )

    # Export the best performing model for deployment
    best_model_path = model.trainer.best
    print(f"\nTraining complete. Best model saved at: {best_model_path}")
    print(f"Exporting best model for deployment...")
    
    export_model = YOLO(best_model_path)
    
    try:
        export_model.export(format='onnx', dynamic=True, simplify=True)
        print("ONNX export successful.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        
    if device == 'cuda':
        try:
            export_model.export(format='engine', device='cuda', workspace=4, dynamic=False)
            print("TensorRT engine export successful.")
        except Exception as e:
            print(f"TensorRT export failed: {e}")

    print(f"\n--- Training and export for '{version}' version complete! ---")
    print(f"Results saved in: '{config['params']['project']}/{config['params']['name']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Mannequin Detection Training Script")
    # ... (parser arguments are the same) ...
    parser.add_argument(
        '--version', 
        type=str, 
        default='toy', 
        choices=['toy', 'real'],
        help="Training configuration: 'toy' for quick testing, 'real' for deployment."
    )
    parser.add_argument(
        '--data',
        type=str,
        default='mannequin_dataset.yaml',
        help="Path to the dataset's .yaml file."
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help="Optional path to custom pre-trained weights (e.g., from DINO-train.py)."
    )
    args = parser.parse_args()
    
    train(version=args.version, data_path=args.data, weights_path=args.weights)