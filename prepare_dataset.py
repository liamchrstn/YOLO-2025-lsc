# prepare_dataset.py
from pathlib import Path
import shutil

def process_labels(source_dir, dest_dir):
    """
    Reads all .txt files from source_dir, filters and remaps class IDs,
    and writes the new files to dest_dir.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Ensure the destination directory exists and is empty
    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    if not source_path.is_dir():
        print(f"Error: Source directory not found: {source_path}")
        return

    # --- This is the key that translates old labels to new ones ---
    # Original Class ID -> New Class ID
    remap_dict = {
        '1': '0',  # mannequins laying -> new class 0
        '2': '1'   # mannequins standing -> new class 1
    }
    
    source_files = list(source_path.glob('*.txt'))
    print(f"Found {len(source_files)} label files in '{source_path}'.")
    
    processed_count = 0
    for source_file in source_files:
        remapped_lines = []
        with open(source_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            original_class_id = parts[0]
            if original_class_id in remap_dict:
                # This is a class we want. Remap it and add it to our list.
                new_class_id = remap_dict[original_class_id]
                remapped_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                remapped_lines.append(remapped_line)
        
        # Write the new file to the destination, but only if it contains labels.
        if remapped_lines:
            dest_file = dest_path / source_file.name
            with open(dest_file, 'w') as f:
                f.writelines(remapped_lines)
            processed_count += 1
    
    print(f"Processing complete. Created {processed_count} clean label files in '{dest_path}'.")


if __name__ == "__main__":
    # Define our source of truth and our clean destination
    SOURCE_TRAIN = r'source_data/original_labels/train'
    SOURCE_VAL = r'source_data/original_labels/val'
    
    DEST_TRAIN = r'dataset/labels/train'
    DEST_VAL = r'dataset/labels/val'
    
    print("--- Preparing Training Dataset ---")
    process_labels(SOURCE_TRAIN, DEST_TRAIN)
    
    print("\n--- Preparing Validation Dataset ---")
    process_labels(SOURCE_VAL, DEST_VAL)
    
    print("\n✅ Dataset preparation finished.")