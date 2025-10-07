import argparse
import torch
from ultralytics import YOLO

CONFIG = {
    'toy': {
        'model_name': 'yolo11n.pt',
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
        'model_name': 'yolo11s.pt',
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

def train(version='toy'):
    """Train and export a YOLOv11 model."""
    if version not in CONFIG:
        print(f"Error: Invalid version '{version}'. Choose from {list(CONFIG.keys())}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting '{version}' version training on device: {device}")
    
    config = CONFIG[version]
    model = YOLO(config['model_name'])

    model.train(
        data='mannequin_dataset.yaml',
        device=device,
        **config['params']
    )

    best_model_path = model.trainer.best
    print(f"Exporting best model for Jetson: {best_model_path}")
    
    export_model = YOLO(best_model_path)
    
    try:
        export_model.export(format='onnx', dynamic=True, simplify=True)
        print("ONNX export successful.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        
    if device == 'cuda':
        try:
            export_model.export(format='engine', device='cuda', workspace=4)
            print("TensorRT engine export successful.")
        except Exception as e:
            print(f"TensorRT export failed: {e}")

    print(f"Training and export for '{version}' version complete!")
    print(f"Results saved in: '{config['params']['project']}/{config['params']['name']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Mannequin Detection Training Script")
    parser.add_argument(
        '--version', 
        type=str, 
        default='toy', 
        choices=['toy', 'real'],
        help="Training configuration: 'toy' for quick testing, 'real' for deployment."
    )
    args = parser.parse_args()
    
    train(args.version)