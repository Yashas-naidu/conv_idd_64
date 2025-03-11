import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
import glob

class IDDTemporalDataset(data.Dataset):
    def __init__(self, sequence_folders, n_frames_input, n_frames_output, 
                 target_size=256, transform=None, include_sequence_info=False):
        '''
        Dataset for IDD temporal data with colored (RGB) frames
        
        Args:
            sequence_folders: list of paths to sequence folders (e.g., paths to *_leftImg8bit folders)
            n_frames_input: number of input frames
            n_frames_output: number of output frames to predict
            target_size: size to resize frames to (square dimensions for the model)
            transform: optional additional transformations
            include_sequence_info: whether to include sequence information in output
        '''
        super(IDDTemporalDataset, self).__init__()
        
        self.sequence_folders = sequence_folders
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.target_size = target_size
        self.transform = transform
        self.include_sequence_info = include_sequence_info
        
        # Process sequences and build index mapping
        self.sequences = self.process_sequences()
        self.index_mapping = self.create_index_mapping()
        self.length = len(self.index_mapping)
    
    def process_sequences(self):
        """Process all sequence folders and identify valid frames"""
        sequences = []
        
        for seq_folder in self.sequence_folders:
            # Extract category and sequence IDs from folder path
            parts = seq_folder.split(os.sep)
            category_id = parts[-2]  # e.g., "00163"
            sequence_id = parts[-1]  # e.g., "155270_leftImg8bit"
            
            # Find all jpeg files in the sequence folder
            frame_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpeg")))
            
            # Only include sequences with enough frames
            if len(frame_files) >= self.n_frames_total:
                sequences.append({
                    'folder': seq_folder,
                    'frames': frame_files,
                    'category': category_id,
                    'sequence': sequence_id,
                    'name': f"{category_id}_{sequence_id}"
                })
            else:
                print(f"Warning: Skipping sequence {seq_folder} with only {len(frame_files)} frames (need {self.n_frames_total})")
        
        print(f"Found {len(sequences)} valid sequences with {self.n_frames_total}+ frames")
        return sequences
    
    def create_index_mapping(self):
        """Create a mapping from dataset indices to (sequence_idx, frame_idx)"""
        index_mapping = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            n_frames = len(sequence['frames'])
            
            # Add indices for all valid frame sequences
            for frame_idx in range(n_frames - self.n_frames_total + 1):
                index_mapping.append((seq_idx, frame_idx))
        
        print(f"Created {len(index_mapping)} valid frame sequences for training/validation")
        return index_mapping
    
    def preprocess_frame(self, frame):
        """Process a frame in RGB"""
        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        frame = cv2.resize(frame, (self.target_size, self.target_size))
        
        # Normalize to 0-1 range
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def __getitem__(self, idx):
        """Get sequence of RGB frames starting at idx"""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        
        # Get sequence and frame indices
        seq_idx, frame_idx = self.index_mapping[idx]
        sequence = self.sequences[seq_idx]
        
        # Get frames for this sequence
        frames = []
        for i in range(self.n_frames_total):
            frame_path = sequence['frames'][frame_idx + i]
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {frame_path}")
            frame = self.preprocess_frame(frame)
            frames.append(frame)
        
        sequence_frames = np.array(frames)
        
        # Split into input and output
        input_frames = sequence_frames[:self.n_frames_input]
        
        if self.n_frames_output > 0:
            output_frames = sequence_frames[self.n_frames_input:self.n_frames_total]
        else:
            output_frames = np.array([])
        
        # Get the last input frame as "frozen" reference
        frozen = input_frames[-1].copy()
        
        # Convert to torch tensors and move channel dimension
        # Move channel dim: (seq_len, height, width, 3) -> (seq_len, 3, height, width)
        input_tensor = torch.from_numpy(input_frames).permute(0, 3, 1, 2).contiguous().float()
        
        if len(output_frames) > 0:
            output_tensor = torch.from_numpy(output_frames).permute(0, 3, 1, 2).contiguous().float()
        else:
            output_tensor = torch.tensor([])
        
        # Extra info for debugging/tracking
        if self.include_sequence_info:
            info = {
                'sequence_name': sequence['name'],
                'category': sequence['category'],
                'sequence_id': sequence['sequence'],
                'frame_idx': frame_idx,
                'frame_paths': sequence['frames'][frame_idx:frame_idx + self.n_frames_total]
            }
            return [idx, output_tensor, input_tensor, frozen, info]
        
        # Format to match the original structure
        # [idx, output_frames, input_frames, frozen, _]
        return [idx, output_tensor, input_tensor, frozen, np.zeros(1)]
    
    def __len__(self):
        return self.length

def find_all_sequence_folders(dataset_root):
    """Find all sequence folders in the dataset"""
    sequence_folders = []
    
    # Walk through the dataset structure
    for category_folder in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category_folder)
        
        if os.path.isdir(category_path):
            # For each category, find all sequence folders
            for sequence_folder in os.listdir(category_path):
                if sequence_folder.endswith('_leftImg8bit'):
                    sequence_path = os.path.join(category_path, sequence_folder)
                    sequence_folders.append(sequence_path)
    
    print(f"Found {len(sequence_folders)} total sequence folders")
    return sequence_folders

def create_idd_datasets(
    dataset_root,
    n_frames_input=10,
    n_frames_output=10,
    target_size=256,
    train_split_ratio=0.8,
    seed=42
):
    """Create train and validation datasets from IDD temporal data"""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find all sequence folders
    all_sequence_folders = find_all_sequence_folders(dataset_root)
    
    # Shuffle and split into train and validation sets
    random.shuffle(all_sequence_folders)
    split_idx = int(len(all_sequence_folders) * train_split_ratio)
    
    train_sequences = all_sequence_folders[:split_idx]
    val_sequences = all_sequence_folders[split_idx:]
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Create train and validation datasets
    train_dataset = IDDTemporalDataset(
        sequence_folders=train_sequences,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        target_size=target_size
    )
    
    val_dataset = IDDTemporalDataset(
        sequence_folders=val_sequences,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        target_size=target_size
    )
    
    return train_dataset, val_dataset

# Main execution block
if __name__ == '__main__':
    # Example usage
    dataset_root = 'idd_temporal_train_4'  # Path to your dataset
    
    # Create the datasets
    train_dataset, val_dataset = create_idd_datasets(
        dataset_root=dataset_root,
        n_frames_input=10,
        n_frames_output=10,
        target_size=256,
        train_split_ratio=0.8
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DataLoaders for train and validation
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,          # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,         # For parallel loading
        pin_memory=True        # Speeds up transfer to GPU
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,          # Adjust based on your GPU memory
        shuffle=False,
        num_workers=4,         # For parallel loading
        pin_memory=True        # Speeds up transfer to GPU
    )
