#!/usr/bin/env python3
"""
Configuration file for dataset paths and parameters
"""

import os

# Dataset paths - update these to match your system
DATASET_PATHS = {
    'idd': "/mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4",
    'japanese': "/mnt/local/gs_datasets/japanese-streets", 
    'vkitti': "/mnt/local/gs_datasets/vkitti"
}

# Dataset ratios (must sum to 1.0)
DATASET_RATIOS = {
    'idd': 0.5,      # 50% IDD
    'japanese': 0.3, # 30% Japanese
    'vkitti': 0.2    # 20% vKITTI
}

# Training parameters
TRAINING_PARAMS = {
    'n_frames_input': 2,
    'n_frames_output': 1,
    'frame_stride': 5,
    'target_size': 256,
    'train_split_ratio': 0.8,
    'motion_threshold': 0.2,
    'batch_size': 70,
    'learning_rate': 1e-4,
    'epochs': 50,
    'random_seed': 1996
}

# DataLoader parameters
DATALOADER_PARAMS = {
    'num_workers': 40,
    'pin_memory': True,
    'shuffle': True
}

def get_dataset_paths():
    """Get dataset paths, checking if they exist"""
    paths = {}
    for name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            paths[name] = path
            print(f"✓ {name.upper()} dataset found: {path}")
        else:
            print(f"✗ {name.upper()} dataset NOT found: {path}")
            print(f"  Please update the path in dataset_config.py")
    
    return paths

def get_dataset_ratios():
    """Get dataset ratios, ensuring they sum to 1.0"""
    total = sum(DATASET_RATIOS.values())
    if abs(total - 1.0) > 1e-6:
        print(f"Warning: Dataset ratios sum to {total}, not 1.0")
        # Normalize ratios
        normalized_ratios = {k: v/total for k, v in DATASET_RATIOS.items()}
        print(f"Normalized ratios: {normalized_ratios}")
        return normalized_ratios
    
    return DATASET_RATIOS

def get_training_params():
    """Get training parameters"""
    return TRAINING_PARAMS.copy()

def get_dataloader_params():
    """Get dataloader parameters"""
    return DATALOADER_PARAMS.copy()

def print_config():
    """Print current configuration"""
    print("Dataset Configuration")
    print("=" * 50)
    
    print("\nDataset Paths:")
    for name, path in DATASET_PATHS.items():
        status = "✓" if os.path.exists(path) else "✗"
        print(f"  {status} {name.upper()}: {path}")
    
    print(f"\nDataset Ratios:")
    for name, ratio in DATASET_RATIOS.items():
        print(f"  {name.upper()}: {ratio*100:.1f}%")
    
    print(f"\nTraining Parameters:")
    for name, value in TRAINING_PARAMS.items():
        print(f"  {name}: {value}")
    
    print(f"\nDataLoader Parameters:")
    for name, value in DATALOADER_PARAMS.items():
        print(f"  {name}: {value}")

if __name__ == "__main__":
    print_config()
    
    # Check if all paths exist
    paths = get_dataset_paths()
    if len(paths) == len(DATASET_PATHS):
        print("\n🎉 All dataset paths are valid!")
    else:
        print(f"\n❌ {len(DATASET_PATHS) - len(paths)} dataset paths are invalid.")
        print("Please update the paths in dataset_config.py")
