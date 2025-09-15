#!/usr/bin/env python3
"""
Script to find available checkpoints for prediction
"""

import os
import glob

def find_checkpoints():
    """Find all available checkpoints in the save_model directory"""
    print("Searching for available checkpoints...")
    print("="*50)
    
    # Look for checkpoints in save_model directory
    save_dir = "./save_model"
    
    if not os.path.exists(save_dir):
        print(f"❌ Save model directory not found: {save_dir}")
        print("Please run training first to generate checkpoints")
        return
    
    # Find all timestamp directories
    timestamp_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    
    if not timestamp_dirs:
        print(f"❌ No timestamp directories found in {save_dir}")
        return
    
    print(f"Found {len(timestamp_dirs)} training runs:")
    print()
    
    for i, timestamp in enumerate(sorted(timestamp_dirs, reverse=True)):
        timestamp_path = os.path.join(save_dir, timestamp)
        print(f"{i+1}. {timestamp}")
        
        # Look for checkpoint files
        checkpoint_files = glob.glob(os.path.join(timestamp_path, "*.pth.tar"))
        
        if checkpoint_files:
            print(f"   📁 Path: {timestamp_path}")
            print(f"   🔑 Checkpoints: {len(checkpoint_files)}")
            
            # Show checkpoint details
            for checkpoint in sorted(checkpoint_files):
                checkpoint_name = os.path.basename(checkpoint)
                file_size = os.path.getsize(checkpoint) / (1024 * 1024)  # MB
                print(f"      • {checkpoint_name} ({file_size:.1f} MB)")
                
                # Try to load checkpoint info
                try:
                    import torch
                    checkpoint_data = torch.load(checkpoint, map_location='cpu')
                    if 'epoch' in checkpoint_data:
                        print(f"        Epoch: {checkpoint_data['epoch']}")
                    if 'config' in checkpoint_data:
                        print(f"        Batch Size: {checkpoint_data['config'].get('batch_size', 'N/A')}")
                        print(f"        Learning Rate: {checkpoint_data['config'].get('learning_rate', 'N/A')}")
                except Exception as e:
                    print(f"        Could not load checkpoint info: {e}")
        else:
            print(f"   📁 Path: {timestamp_path}")
            print(f"   ❌ No checkpoint files found")
        
        print()
    
    # Provide usage instructions
    print("="*50)
    print("To use a checkpoint for prediction:")
    print("1. Copy the full path to the checkpoint file")
    print("2. Update CHECKPOINT_PATH in predict_mixed.py")
    print("3. Run: python predict_mixed.py")
    print()
    print("Example checkpoint path:")
    if timestamp_dirs:
        example_dir = timestamp_dirs[0]
        example_checkpoints = glob.glob(os.path.join(save_dir, example_dir, "*.pth.tar"))
        if example_checkpoints:
            print(f"   {example_checkpoints[0]}")

if __name__ == "__main__":
    find_checkpoints()
