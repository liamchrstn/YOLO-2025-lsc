"""
DINO-train.py: Self-supervised pre-training script for YOLO models using Lightly and DINOv3.
This script uses the official lightly_train Python API and a Hugging Face Hub model ID.
"""
import os
import argparse
import yaml
import lightly_train

# Configuration now uses the official Hugging Face Hub ID
CONFIG = {
    'toy': {
        'model_yaml': 'ultralytics/yolo11n.yaml',
        # --- THE SOLUTION ---
        # Use the official Hugging Face Hub ID for the teacher model.
        'teacher_model_identifier': 'facebook/dino-vits8',
        'epochs': 20,
        'batch_size': 32,
        'out_dir': 'runs/pretrain_dino_toy'
    },
    'real': {
        'model_yaml': 'ultralytics/yolo11s.yaml',
        # Assuming the ViT-Base model for the 'real' version
        'teacher_model_identifier': 'facebook/dino-vitb8',
        'epochs': 100,
        'batch_size': 16,
        'out_dir': 'runs/pretrain_dino_real'
    }
}

def pretrain_with_dino(version='toy', data_yaml='mannequin_dataset.yaml'):
    """
    Constructs and runs the training using the lightly_train Python API.
    """
    if version not in CONFIG:
        print(f"Error: Invalid version '{version}'. Choose from {list(CONFIG.keys())}")
        return

    config = CONFIG[version]
    
    print(f"--- Starting DINOv3 Self-Supervised Pre-training ('{version}' version) ---")

    try:
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        image_path = os.path.abspath(data_config['train'])
        print(f"Using training images from: {image_path}")
    except Exception as e:
        print(f"Error: Could not read '{data_yaml}': {e}")
        return

    print("\nStarting lightly_train.train() with the following configuration:")
    
    try:
        lightly_train.train(
            out=config['out_dir'],
            data=image_path,
            model=config['model_yaml'],
            method='distillation',
            method_args={
                # Pass the Hub ID directly to the 'teacher' argument.
                'teacher': config['teacher_model_identifier']
            },
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            accelerator='auto',
            devices=1,
            num_workers=0,
            overwrite=True
        )
    except Exception as e:
        print(f"\n--- An error occurred during training ---")
        import traceback
        traceback.print_exc()
        return

    print(f"\n--- DINOv3 Pre-training ('{version}' version) Finished ---")
    print(f"Pre-trained model saved in: '{config['out_dir']}/exported_models/last.pt'")
    print("You can now use this weights file for supervised fine-tuning.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO Pre-training script for YOLO models.")
    parser.add_argument(
        '--data', 
        type=str, 
        default='mannequin_dataset.yaml', 
        help='Path to your data.yaml file.'
    )
    parser.add_argument(
        '--version', 
        type=str, 
        default='toy', 
        choices=['toy', 'real'],
        help="Pre-training configuration: 'toy' for quick testing, 'real' for full pre-training."
    )
    args = parser.parse_args()
    
    pretrain_with_dino(args.version, args.data)