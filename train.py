import os
import argparse
import yaml
import torch
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='model_config.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--device', default='mps', help='cuda device, i.e. 0 or 0,1,2,3 or cpu or mps for M1 Mac')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    return parser.parse_args()

def prepare_dataset(data_cfg):
    """
    Prepares the dataset by checking label files and splitting into train/val sets
    """
    print("Checking dataset structure...")
    
    # Read classes file
    try:
        with open(os.path.join('data', 'label', 'classes.txt'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"Found classes: {classes}")
    except FileNotFoundError:
        print("Warning: classes.txt not found, assuming 'fire' as the only class.")
        classes = ['fire']
    
    # Check for matching image and label files
    image_dir = Path('datav1/image')
    label_dir = Path('datav1/label')
    
    image_files = list(image_dir.glob('*.jpeg'))
    image_files += list(image_dir.glob('*.jpg'))
    image_files += list(image_dir.glob('*.png'))
    image_files.sort()
    label_files = sorted([f for f in label_dir.glob('*.txt') if f.stem != 'classes'])
    
    print(f"Found {len(image_files)} images and {len(label_files)} label files")
    
    # Validate matching pairs
    image_stems = [f.stem for f in image_files]
    label_stems = [f.stem for f in label_files]
    
    matches = set(image_stems).intersection(set(label_stems))
    print(f"Found {len(matches)} matching image-label pairs")
    
    if len(matches) == 0:
        print("Error: No matching image-label pairs found!")
        sys.exit(1)
    
    # Create a new dataset.yaml with the correct paths
    dataset_yaml = {
        'path': os.path.abspath('data'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    # Create training split directories
    os.makedirs(os.path.join('data', 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join('data', 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join('data', 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join('data', 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join('data', 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join('data', 'labels', 'test'), exist_ok=True)
    
    # Create symbolic links or copy files to the split directories
    # Using 80/10/10 split for train/val/test
    import random
    import shutil
    
    # Shuffle the matches
    match_list = list(matches)
    random.shuffle(match_list)
    
    # Calculate split indices
    train_end = int(0.8 * len(match_list))
    val_end = int(0.9 * len(match_list))
    
    # Split into train/val/test
    train_set = match_list[:train_end]
    val_set = match_list[train_end:val_end]
    test_set = match_list[val_end:]
    
    print(f"Split dataset: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
    
    # Copy files to split directories
    def copy_files(file_stems, target_set):
        for stem in file_stems:
            # Find image file with this stem
            for ext in ['.jpeg', '.jpg', '.png']:
                img_file = image_dir / f"{stem}{ext}"
                if img_file.exists():
                    shutil.copy(img_file, os.path.join('data', 'images', target_set, f"{stem}{ext}"))
                    break
            
            # Copy label file
            label_file = label_dir / f"{stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, os.path.join('data', 'labels', target_set, f"{stem}.txt"))
    
    copy_files(train_set, 'train')
    copy_files(val_set, 'val')
    copy_files(test_set, 'test')
    
    # Save dataset yaml file
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    
    return 'dataset.yaml'

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create the YOLOv5 directory if it doesn't exist
    if not os.path.exists('yolov5'):
        print("Cloning YOLOv5 repository...")
        os.system('git clone https://github.com/ultralytics/yolov5.git')
    
    # Read data configuration
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Prepare dataset - restructure if needed
    dataset_yaml = prepare_dataset(data_cfg)
    
    # M1 Mac specific settings
    device_arg = "--device mps" if args.device == "mps" else f"--device {args.device}"
    
    # Add MPS support patch for M1 Macs (inside yolov5/utils/torch_utils.py)
    torch_utils_path = os.path.join('yolov5', 'utils', 'torch_utils.py')
    if os.path.exists(torch_utils_path) and args.device == 'mps':
        print("Applying patch for M1 Mac MPS support...")
        with open(torch_utils_path, 'r') as f:
            content = f.read()
        
        # Check if patch is needed
        if 'mps' not in content:
            patched_content = content.replace(
                "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
                "device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))"
            )
            
            # Add MPS to several other places in the file
            patched_content = patched_content.replace(
                "TORCH_1_11 = torch.__version__ >= '1.11.0'",
                "TORCH_1_11 = torch.__version__ >= '1.11.0'\nTORCH_2_0 = torch.__version__ >= '2.0.0'"
            )
            
            with open(torch_utils_path, 'w') as f:
                f.write(patched_content)
            
            print("Patch applied successfully")
    
    # Execute training command
    print("Starting training...")
    cmd = (f"cd yolov5 && python3 train.py "
       f"--img {args.img_size[0]} "
       f"--batch {args.batch_size} "
       f"--epochs {args.epochs} "
       f"--data ../{dataset_yaml} "
       f"--weights {args.weights} "
       f"{device_arg} "
       f"--workers {args.workers} "
       f"--cache {'images' if args.cache else ''}")
    
    print(f"Executing: {cmd}")
    os.system(cmd)
    
    # Export to optimized models for deployment
    print("Exporting model for deployment...")
    weights_path = Path('yolov5/runs/train/exp/weights/best.pt')
    if weights_path.exists():
        # Export to ONNX format for Jetson Nano and CoreML for M1 Mac
        os.system(f"cd yolov5 && python export.py --weights {weights_path} --include onnx coreml --device {args.device}")
        print(f"Model exported to ONNX at {weights_path.with_suffix('.onnx')}")
        print(f"Model exported to CoreML at {weights_path.with_suffix('.mlmodel')}")
    else:
        print("Training completed but best.pt not found. Check the training logs.")

if __name__ == "__main__":
    main()