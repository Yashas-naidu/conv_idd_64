#!/usr/bin/env python3
"""
Setup script for prediction configuration
"""

import os
import glob

def setup_prediction():
    """Setup prediction configuration"""
    print("Mixed Dataset Prediction Setup")
    print("="*50)
    
    # Find available checkpoints
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
    
    print("Available training runs:")
    print()
    
    for i, timestamp in enumerate(sorted(timestamp_dirs, reverse=True)):
        timestamp_path = os.path.join(save_dir, timestamp)
        checkpoint_files = glob.glob(os.path.join(timestamp_path, "*.pth.tar"))
        
        if checkpoint_files:
            print(f"{i+1}. {timestamp}")
            print(f"   Checkpoints: {len(checkpoint_files)}")
            for checkpoint in sorted(checkpoint_files):
                checkpoint_name = os.path.basename(checkpoint)
                print(f"      • {checkpoint_name}")
            print()
    
    # Let user select
    try:
        choice = int(input("Select training run (number): ")) - 1
        if 0 <= choice < len(timestamp_dirs):
            selected_timestamp = sorted(timestamp_dirs, reverse=True)[choice]
            selected_path = os.path.join(save_dir, selected_timestamp)
            
            # Find checkpoints in selected directory
            checkpoint_files = glob.glob(os.path.join(selected_path, "*.pth.tar"))
            
            if checkpoint_files:
                print(f"\nSelected: {selected_timestamp}")
                print("Available checkpoints:")
                
                for i, checkpoint in enumerate(sorted(checkpoint_files)):
                    checkpoint_name = os.path.basename(checkpoint)
                    print(f"  {i+1}. {checkpoint_name}")
                
                checkpoint_choice = int(input("Select checkpoint (number): ")) - 1
                if 0 <= checkpoint_choice < len(checkpoint_files):
                    selected_checkpoint = sorted(checkpoint_files)[checkpoint_choice]
                    
                    print(f"\n✅ Selected checkpoint: {selected_checkpoint}")
                    
                    # Update predict_mixed.py
                    update_prediction_script(selected_checkpoint)
                    
                else:
                    print("❌ Invalid checkpoint selection")
            else:
                print(f"❌ No checkpoints found in {selected_timestamp}")
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Please enter a valid number")
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled")

def update_prediction_script(checkpoint_path):
    """Update the checkpoint path in predict_mixed.py"""
    script_path = "./predict_mixed.py"
    
    if not os.path.exists(script_path):
        print(f"❌ predict_mixed.py not found: {script_path}")
        return
    
    try:
        # Read the script
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Update checkpoint path
        old_path = 'CHECKPOINT_PATH = "./save_model/2024-01-XXTXX-XX-XX/checkpoint.pth.tar"'
        new_path = f'CHECKPOINT_PATH = "{checkpoint_path}"'
        
        if old_path in content:
            content = content.replace(old_path, new_path)
            
            # Write back
            with open(script_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated {script_path} with checkpoint path")
            print(f"   New path: {checkpoint_path}")
            
            # Show next steps
            print("\n🚀 Next steps:")
            print("1. Run prediction: python predict_mixed.py")
            print("2. Check results in ./prediction_results/")
            
        else:
            print("❌ Could not find checkpoint path line in script")
            
    except Exception as e:
        print(f"❌ Error updating script: {e}")

if __name__ == "__main__":
    setup_prediction()
