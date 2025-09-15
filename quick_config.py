#!/usr/bin/env python3
"""
Quick configuration viewer and modifier for train_mixed.py
"""

def show_current_config():
    """Show the current hardcoded configuration"""
    print("Current Hardcoded Configuration in train_mixed.py:")
    print("=" * 50)
    
    print("\n📊 Training Parameters:")
    print("  Batch Size: 30")
    print("  Learning Rate: 1e-4")
    print("  Epochs: 1")
    print("  Motion Threshold: 0.2")
    print("  Frames Input: 2")
    print("  Frames Output: 1")
    print("  Seed: 1996")
    
    print("\n🗂️ Dataset Paths:")
    print("  IDD: /mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4")
    print("  Japanese: /mnt/local/gs_datasets/japanese-streets")
    print("  vKITTI: /mnt/local/gs_datasets/vkitti")
    
    print("\n⚖️ Dataset Ratios:")
    print("  IDD: 50%")
    print("  Japanese: 30%")
    print("  vKITTI: 20%")

def show_how_to_modify():
    """Show how to modify the configuration"""
    print("\n" + "=" * 50)
    print("How to Modify Configuration:")
    print("=" * 50)
    
    print("\n1. Open train_mixed.py in your editor")
    print("2. Find the TRAINING_CONFIG dictionary (around line 30)")
    print("3. Modify the values you want to change:")
    print("   - batch_size: Change from 30 to your desired batch size")
    print("   - epochs: Change from 1 to your desired number of epochs")
    print("   - learning_rate: Change from 1e-4 to your desired learning rate")
    print("   - motion_threshold: Adjust motion detection sensitivity")
    
    print("\n4. Find the DATASET_PATHS dictionary (around line 40)")
    print("5. Update the paths if your datasets are in different locations")
    
    print("\n6. Find the DATASET_RATIOS dictionary (around line 50)")
    print("7. Adjust the ratios if you want different proportions")
    print("   - Make sure the ratios sum to 1.0")
    
    print("\n8. Save the file and run: python train_mixed.py")

def show_example_modifications():
    """Show example modifications"""
    print("\n" + "=" * 50)
    print("Example Modifications:")
    print("=" * 50)
    
    print("\n📈 For longer training:")
    print("  'epochs': 50")
    print("  'batch_size': 70")
    
    print("\n🔍 For more sensitive motion detection:")
    print("  'motion_threshold': 0.1")
    
    print("\n⚡ For faster training:")
    print("  'batch_size': 16")
    print("  'learning_rate': 5e-4")
    
    print("\n🎯 For different dataset balance:")
    print("  'idd': 0.6,      # 60% IDD")
    print("  'japanese': 0.25, # 25% Japanese")
    print("  'vkitti': 0.15    # 15% vKITTI")

if __name__ == "__main__":
    show_current_config()
    show_how_to_modify()
    show_example_modifications()
    
    print("\n" + "=" * 50)
    print("💡 Tip: After modifying train_mixed.py, you can run:")
    print("   python train_mixed.py")
    print("   No command line arguments needed!")
    print("=" * 50)
