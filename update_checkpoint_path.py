#!/usr/bin/env python3
"""
Simple script to update the hardcoded checkpoint path in predict_mixed.py
"""

import os
import glob

def update_checkpoint_path():
    """Update the hardcoded checkpoint path"""
    print("Update Checkpoint Path in predict_mixed.py")
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
                    update_script(selected_checkpoint)
                    
                else:
                    print("❌ Invalid checkpoint selection")
            else:
                print(f"❌ No checkpoints found in {selected_timestamp}")
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Please enter a valid number")
    except KeyboardInterrupt:
        print("\n❌ Update cancelled")

def update_script(checkpoint_path):
    """Update the checkpoint path in predict_mixed.py"""
    script_path = "./predict_mixed.py"
    
    if not os.path.exists(script_path):
        print(f"❌ predict_mixed.py not found: {script_path}")
        return
    
    try:
        # Read the script
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Find and update the checkpoint path line
        old_line = 'CHECKPOINT_PATH = "./save_model/2024-01-XXTXX-XX-XX/checkpoint.pth.tar"  # HARDCODED - Update this path to your actual checkpoint'
        new_line = f'CHECKPOINT_PATH = "{checkpoint_path}"  # HARDCODED - Updated automatically'
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            
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
            print("Please check the script manually")
            
    except Exception as e:
        print(f"❌ Error updating script: {e}")

def show_current_path():
    """Show the current checkpoint path in the script"""
    script_path = "./predict_mixed.py"
    
    if not os.path.exists(script_path):
        print(f"❌ predict_mixed.py not found: {script_path}")
        return
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Find the checkpoint path line
        for line in content.split('\n'):
            if line.strip().startswith('CHECKPOINT_PATH ='):
                print("Current checkpoint path in predict_mixed.py:")
                print(f"  {line.strip()}")
                return
        
        print("❌ Could not find CHECKPOINT_PATH in script")
        
    except Exception as e:
        print(f"❌ Error reading script: {e}")

if __name__ == "__main__":
    print("Checkpoint Path Updater")
    print("="*50)
    
    # Show current path
    show_current_path()
    print()
    
    # Ask what to do
    print("What would you like to do?")
    print("1. Update checkpoint path")
    print("2. Show current path only")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            update_checkpoint_path()
        elif choice == "2":
            print("Current path shown above.")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
