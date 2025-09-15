#!/usr/bin/env python3
"""
Test script for the mixed dataset to verify it can load data from all three sources
"""

import os
import sys
from mixed_dataset import create_mixed_datasets, visualize_mixed_samples

def test_mixed_dataset():
    """Test the mixed dataset creation and loading"""
    
    # Get dataset paths from config
    from dataset_config import get_dataset_paths, get_dataset_ratios
    dataset_paths = get_dataset_paths()
    dataset_ratios = get_dataset_ratios()
    
    if len(dataset_paths) != 3:
        print("❌ Not all dataset paths are valid. Please check dataset_config.py")
        return False
    
    idd_root = dataset_paths['idd']
    japanese_root = dataset_paths['japanese']
    vkitti_root = dataset_paths['vkitti']
    
    print("Testing mixed dataset creation...")
    print(f"IDD root: {idd_root}")
    print(f"Japanese root: {japanese_root}")
    print(f"vKITTI root: {vkitti_root}")
    
    # Check if paths exist
    for name, path in [("IDD", idd_root), ("Japanese", japanese_root), ("vKITTI", vkitti_root)]:
        if os.path.exists(path):
            print(f"✓ {name} dataset path exists: {path}")
        else:
            print(f"✗ {name} dataset path NOT found: {path}")
            print(f"  Please update the path in this script")
            return False
    
    try:
        # Create mixed datasets
        print("\nCreating mixed datasets...")
        train_dataset, val_dataset = create_mixed_datasets(
            idd_root=idd_root,
            japanese_root=japanese_root,
            vkitti_root=vkitti_root,
            n_frames_input=2,
            n_frames_output=1,
            frame_stride=5,
            target_size=256,
            train_split_ratio=0.8,
            seed=42,
            motion_threshold=0.1,
            idd_ratio=dataset_ratios['idd'],
            japanese_ratio=dataset_ratios['japanese'],
            vkitti_ratio=dataset_ratios['vkitti']
        )
        
        print(f"\n✓ Successfully created mixed datasets!")
        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Validation dataset size: {len(val_dataset)}")
        
        # Test loading a few samples
        print("\nTesting data loading...")
        if len(train_dataset) > 0:
            # Test first sample
            sample = train_dataset[0]
            idx, output_tensor, input_tensor, frozen, info = sample
            
            print(f"✓ Successfully loaded sample 0")
            print(f"  Dataset: {info['dataset']}")
            print(f"  Sequence: {info['sequence_name']}")
            print(f"  Input tensor shape: {input_tensor.shape}")
            print(f"  Output tensor shape: {output_tensor.shape}")
            print(f"  Frozen tensor shape: {frozen.shape}")
            
            # Test a few more samples
            num_test_samples = min(5, len(train_dataset))
            print(f"\nTesting {num_test_samples} more samples...")
            
            for i in range(1, num_test_samples):
                try:
                    sample = train_dataset[i]
                    _, _, _, _, info = sample
                    print(f"  Sample {i}: {info['dataset']} - {info['sequence_name']}")
                except Exception as e:
                    print(f"  ✗ Error loading sample {i}: {e}")
                    return False
            
            print("✓ All test samples loaded successfully!")
            
            # Show dataset composition
            print(f"\nDataset composition:")
            datasets = {}
            for i in range(min(100, len(train_dataset))):  # Check first 100 samples
                sample = train_dataset[i]
                _, _, _, _, info = sample
                dataset_type = info['dataset']
                datasets[dataset_type] = datasets.get(dataset_type, 0) + 1
            
            total_checked = sum(datasets.values())
            for dataset_type, count in datasets.items():
                percentage = (count / total_checked) * 100
                print(f"  {dataset_type.upper()}: {count} samples ({percentage:.1f}%)")
            
            return True
            
        else:
            print("✗ Train dataset is empty!")
            return False
            
    except Exception as e:
        print(f"✗ Error creating mixed dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_datasets():
    """Test each dataset individually to identify issues"""
    print("\n" + "="*50)
    print("Testing individual datasets...")
    
    from mixed_dataset import find_idd_sequence_folders, find_japanese_sequence_folders, find_vkitti_sequence_folders
    from dataset_config import get_dataset_paths
    
    dataset_paths = get_dataset_paths()
    
    # Test IDD
    print("\n--- Testing IDD Dataset ---")
    try:
        idd_sequences = find_idd_sequence_folders(dataset_paths['idd'])
        print(f"Found {len(idd_sequences)} IDD sequences")
        if len(idd_sequences) > 0:
            print(f"First sequence: {idd_sequences[0]}")
    except Exception as e:
        print(f"Error with IDD: {e}")
    
    # Test Japanese
    print("\n--- Testing Japanese Dataset ---")
    try:
        japanese_sequences = find_japanese_sequence_folders(dataset_paths['japanese'])
        print(f"Found {len(japanese_sequences)} Japanese sequences")
        if len(japanese_sequences) > 0:
            print(f"First sequence: {japanese_sequences[0]}")
    except Exception as e:
        print(f"Error with Japanese: {e}")
    
    # Test vKITTI
    print("\n--- Testing vKITTI Dataset ---")
    try:
        vkitti_sequences = find_vkitti_sequence_folders(dataset_paths['vkitti'])
        print(f"Found {len(vkitti_sequences)} vKITTI sequences")
        if len(vkitti_sequences) > 0:
            print(f"First sequence: {vkitti_sequences[0]}")
    except Exception as e:
        print(f"Error with vKITTI: {e}")

if __name__ == "__main__":
    print("Mixed Dataset Test Script")
    print("="*50)
    
    # First test individual datasets
    test_individual_datasets()
    
    # Then test the combined mixed dataset
    print("\n" + "="*50)
    success = test_mixed_dataset()
    
    if success:
        print("\n🎉 All tests passed! The mixed dataset is working correctly.")
        print("\nYou can now use it for training with:")
        print("python train_mixed.py")
        print("\nOr check the configuration with:")
        print("python train_mixed.py --config_check")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dataset paths are correct and accessible.")
