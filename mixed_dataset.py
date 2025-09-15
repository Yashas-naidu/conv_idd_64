import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
import random
import glob
import matplotlib.pyplot as plt

class MixedTemporalDataset(data.Dataset):
    def __init__(self, idd_sequences, japanese_sequences, vkitti_sequences, 
                 n_frames_input=2, n_frames_output=1, frame_stride=5, 
                 target_size=256, transform=None, include_sequence_info=False,
                 motion_threshold=0.01, idd_ratio=0.5, japanese_ratio=0.3, vkitti_ratio=0.2):
        '''
        Mixed dataset combining IDD, Japanese streets, and vKITTI with specified ratios
        
        Args:
            idd_sequences: list of IDD sequence folders
            japanese_sequences: list of Japanese streets sequence folders
            vkitti_sequences: list of vKITTI sequence folders
            n_frames_input: number of input frames (2)
            n_frames_output: number of output frames to predict (1)
            frame_stride: number of frames between consecutive input/output frames (5)
            target_size: size to resize frames to (square dimensions for the model)
            transform: optional additional transformations
            include_sequence_info: whether to include sequence information in output
            motion_threshold: threshold for motion detection
            idd_ratio: proportion of IDD data (0.5)
            japanese_ratio: proportion of Japanese data (0.3)
            vkitti_ratio: proportion of vKITTI data (0.2)
        '''
        super(MixedTemporalDataset, self).__init__()
        
        self.idd_sequences = idd_sequences
        self.japanese_sequences = japanese_sequences
        self.vkitti_sequences = vkitti_sequences
        
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.frame_stride = frame_stride
        self.n_frames_total = (n_frames_input - 1) * frame_stride + frame_stride + 1
        self.target_size = target_size
        self.transform = transform
        self.include_sequence_info = include_sequence_info
        self.motion_threshold = motion_threshold
        
        # Dataset ratios
        self.idd_ratio = idd_ratio
        self.japanese_ratio = japanese_ratio
        self.vkitti_ratio = vkitti_ratio
        
        # Process sequences and build index mapping
        self.sequences = self.process_sequences()
        self.index_mapping = self.create_index_mapping()
        self.length = len(self.index_mapping)
        
        print(f"Mixed dataset created with {len(self.sequences)} total sequences")
        print(f"IDD: {len([s for s in self.sequences if s['dataset'] == 'idd'])} sequences")
        print(f"Japanese: {len([s for s in self.sequences if s['dataset'] == 'japanese'])} sequences")
        print(f"vKITTI: {len([s for s in self.sequences if s['dataset'] == 'vkitti'])} sequences")
    
    def detect_motion(self, frames):
        """Detect motion between frames by calculating pixel differences"""
        small_size = (64, 64)
        total_pixels = small_size[0] * small_size[1]
        
        # Load and preprocess the first frame
        prev_frame = cv2.imread(frames[0])
        if prev_frame is None:
            return False
        prev_frame = cv2.resize(prev_frame, small_size)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Compare consecutive frames
        for i in range(1, len(frames)):
            curr_frame = cv2.imread(frames[i])
            if curr_frame is None:
                return False
            curr_frame = cv2.resize(curr_frame, small_size)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute absolute difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            changed_pixels = np.sum(thresh) / 255.0
            fraction_changed = changed_pixels / total_pixels
            
            if fraction_changed > self.motion_threshold:
                return True
            
            prev_frame = curr_frame
        
        return False
    
    def process_sequences(self):
        """Process all sequence folders from different datasets"""
        sequences = []
        
        # Process IDD sequences
        for seq_folder in self.idd_sequences:
            parts = seq_folder.split(os.sep)
            category_id = parts[-2]
            sequence_id = parts[-1]
            
            frame_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpeg")))
            
            if len(frame_files) >= self.n_frames_total and self.detect_motion(frame_files):
                sequences.append({
                    'folder': seq_folder,
                    'frames': frame_files,
                    'category': category_id,
                    'sequence': sequence_id,
                    'name': f"IDD_{category_id}_{sequence_id}",
                    'dataset': 'idd'
                })
        
        # Process Japanese streets sequences
        for seq_folder in self.japanese_sequences:
            parts = seq_folder.split(os.sep)
            group_id = parts[-2]
            set_id = parts[-1]
            
            frame_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpg")))
            
            if len(frame_files) >= self.n_frames_total and self.detect_motion(frame_files):
                sequences.append({
                    'folder': seq_folder,
                    'frames': frame_files,
                    'category': group_id,
                    'sequence': set_id,
                    'name': f"JAPANESE_{group_id}_{set_id}",
                    'dataset': 'japanese'
                })
        
        # Process vKITTI sequences
        for seq_folder in self.vkitti_sequences:
            parts = seq_folder.split(os.sep)
            scene_id = parts[-4]
            condition = parts[-3]
            camera_id = parts[-1]
            
            frame_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpg")))
            
            if len(frame_files) >= self.n_frames_total and self.detect_motion(frame_files):
                sequences.append({
                    'folder': seq_folder,
                    'frames': frame_files,
                    'category': scene_id,
                    'sequence': f"{condition}_{camera_id}",
                    'name': f"VKITTI_{scene_id}_{condition}_{camera_id}",
                    'dataset': 'vkitti'
                })
        
        return sequences
    
    def create_index_mapping(self):
        """Create a mapping from dataset indices to (sequence_idx, frame_idx)"""
        index_mapping = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            n_frames = len(sequence['frames'])
            
            # Add indices for all valid frame sequences with stride
            for frame_idx in range(n_frames - self.n_frames_total + 1):
                index_mapping.append((seq_idx, frame_idx))
        
        return index_mapping
    
    def preprocess_frame(self, frame):
        """Process a frame in RGB"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.target_size, self.target_size))
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def __getitem__(self, idx):
        """Get sequence of RGB frames starting at idx with stride"""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        
        seq_idx, frame_idx = self.index_mapping[idx]
        sequence = self.sequences[seq_idx]
        
        # Get frames with stride
        frames = []
        for i in range(self.n_frames_input + self.n_frames_output):
            frame_pos = frame_idx + i * self.frame_stride
            frame_path = sequence['frames'][frame_pos]
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {frame_path}")
            frame = self.preprocess_frame(frame)
            frames.append(frame)
        
        sequence_frames = np.array(frames)
        
        # Split into input and output
        input_frames = sequence_frames[:self.n_frames_input]
        output_frames = sequence_frames[self.n_frames_input:]
        
        # Get the last input frame as "frozen" reference
        frozen = input_frames[-1].copy()
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_frames).permute(0, 3, 1, 2).contiguous().float()
        output_tensor = torch.from_numpy(output_frames).permute(0, 3, 1, 2).contiguous().float()
        
        if self.include_sequence_info:
            info = {
                'sequence_name': sequence['name'],
                'dataset': sequence['dataset'],
                'category': sequence['category'],
                'sequence_id': sequence['sequence'],
                'frame_idx': frame_idx,
                'frame_paths': [sequence['frames'][frame_idx + i * self.frame_stride] 
                                for i in range(self.n_frames_input + self.n_frames_output)]
            }
            return [idx, output_tensor, input_tensor, frozen, info]
        
        return [idx, output_tensor, input_tensor, frozen, np.zeros(1)]
    
    def __len__(self):
        return self.length

def find_idd_sequence_folders(dataset_root):
    """Find all IDD sequence folders"""
    sequence_folders = []
    
    for category_folder in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category_folder)
        
        if os.path.isdir(category_path):
            for sequence_folder in os.listdir(category_path):
                if sequence_folder.endswith('_leftImg8bit'):
                    sequence_path = os.path.join(category_path, sequence_folder)
                    sequence_folders.append(sequence_path)
    
    print(f"Found {len(sequence_folders)} IDD sequence folders")
    return sequence_folders

def find_japanese_sequence_folders(dataset_root):
    """Find all Japanese streets sequence folders"""
    sequence_folders = []
    
    frames_dir = os.path.join(dataset_root, "frames")
    if not os.path.exists(frames_dir):
        print(f"Japanese frames directory not found: {frames_dir}")
        return sequence_folders
    
    for group_folder in os.listdir(frames_dir):
        group_path = os.path.join(frames_dir, group_folder)
        
        if os.path.isdir(group_path):
            for set_folder in os.listdir(group_path):
                set_path = os.path.join(group_path, set_folder)
                if os.path.isdir(set_path):
                    # Check if there are jpg files
                    jpg_files = glob.glob(os.path.join(set_path, "*.jpg"))
                    if len(jpg_files) > 0:
                        sequence_folders.append(set_path)
    
    print(f"Found {len(sequence_folders)} Japanese sequence folders")
    return sequence_folders

def find_vkitti_sequence_folders(dataset_root):
    """Find all vKITTI sequence folders"""
    sequence_folders = []
    
    for scene_folder in os.listdir(dataset_root):
        scene_path = os.path.join(dataset_root, scene_folder)
        
        if os.path.isdir(scene_path):
            for condition_folder in os.listdir(scene_path):
                condition_path = os.path.join(scene_path, condition_folder)
                
                if os.path.isdir(condition_path):
                    frames_path = os.path.join(condition_path, "frames", "rgb")
                    if os.path.exists(frames_path):
                        for camera_folder in os.listdir(frames_path):
                            camera_path = os.path.join(frames_path, camera_folder)
                            if os.path.isdir(camera_path):
                                # Check if there are jpg files
                                jpg_files = glob.glob(os.path.join(camera_path, "*.jpg"))
                                if len(jpg_files) > 0:
                                    sequence_folders.append(camera_path)
    
    print(f"Found {len(sequence_folders)} vKITTI sequence folders")
    return sequence_folders

def create_mixed_datasets(
    idd_root,
    japanese_root,
    vkitti_root,
    n_frames_input=2,
    n_frames_output=1,
    frame_stride=5,
    target_size=256,
    train_split_ratio=0.8,
    seed=None,
    motion_threshold=0.01,
    idd_ratio=0.5,
    japanese_ratio=0.3,
    vkitti_ratio=0.2
):
    """Create train and validation datasets from mixed temporal data"""
    random.seed(seed)
    
    # Find all sequence folders from each dataset
    idd_sequences = find_idd_sequence_folders(idd_root)
    japanese_sequences = find_japanese_sequence_folders(japanese_root)
    vkitti_sequences = find_vkitti_sequence_folders(vkitti_root)
    
    # Shuffle each dataset separately
    random.shuffle(idd_sequences)
    random.shuffle(japanese_sequences)
    random.shuffle(vkitti_sequences)
    
    # Split into train and validation sets for each dataset
    idd_split_idx = int(len(idd_sequences) * train_split_ratio)
    japanese_split_idx = int(len(japanese_sequences) * train_split_ratio)
    vkitti_split_idx = int(len(vkitti_sequences) * train_split_ratio)
    
    train_idd = idd_sequences[:idd_split_idx]
    val_idd = idd_sequences[idd_split_idx:]
    
    train_japanese = japanese_sequences[:japanese_split_idx]
    val_japanese = japanese_sequences[japanese_split_idx:]
    
    train_vkitti = vkitti_sequences[:vkitti_split_idx]
    val_vkitti = vkitti_sequences[vkitti_split_idx:]
    
    print(f"Train sequences - IDD: {len(train_idd)}, Japanese: {len(train_japanese)}, vKITTI: {len(train_vkitti)}")
    print(f"Val sequences - IDD: {len(val_idd)}, Japanese: {len(val_japanese)}, vKITTI: {len(val_vkitti)}")
    
    # Create train and validation datasets
    train_dataset = MixedTemporalDataset(
        idd_sequences=train_idd,
        japanese_sequences=train_japanese,
        vkitti_sequences=train_vkitti,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        frame_stride=frame_stride,
        target_size=target_size,
        motion_threshold=motion_threshold,
        idd_ratio=idd_ratio,
        japanese_ratio=japanese_ratio,
        vkitti_ratio=vkitti_ratio,
        include_sequence_info=True
    )
    
    val_dataset = MixedTemporalDataset(
        idd_sequences=val_idd,
        japanese_sequences=val_japanese,
        vkitti_sequences=val_vkitti,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        frame_stride=frame_stride,
        target_size=target_size,
        motion_threshold=motion_threshold,
        idd_ratio=idd_ratio,
        japanese_ratio=japanese_ratio,
        vkitti_ratio=vkitti_ratio,
        include_sequence_info=True
    )
    
    return train_dataset, val_dataset

def visualize_mixed_samples(dataset, num_samples=10):
    """Visualize samples from the mixed dataset"""
    if len(dataset) == 0:
        print("Dataset is empty after motion filtering.")
        return
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        sample = dataset[idx]
        _, output_tensor, input_tensor, _, info = sample
        
        input_frames = input_tensor.numpy().transpose(0, 2, 3, 1)
        output_frames = output_tensor.numpy().transpose(0, 2, 3, 1)
        
        fig, axes = plt.subplots(1, dataset.n_frames_input + dataset.n_frames_output, figsize=(15, 5))
        fig.suptitle(f"Sample {idx} - {info['dataset'].upper()}: {info['sequence_name']}")
        
        for i in range(dataset.n_frames_input):
            axes[i].imshow(input_frames[i])
            axes[i].set_title(f"Input Frame {(i * dataset.frame_stride)}")
            axes[i].axis('off')
        
        for i in range(dataset.n_frames_output):
            axes[dataset.n_frames_input + i].imshow(output_frames[i])
            axes[dataset.n_frames_input + i].set_title(f"Output Frame {(dataset.n_frames_input + i) * dataset.frame_stride}")
            axes[dataset.n_frames_input + i].axis('off')
        
        plt.show()

if __name__ == '__main__':
    # Example usage
    idd_root = "/mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4"
    japanese_root = "/mnt/local/gs_datasets/japanese-streets"
    vkitti_root = "/mnt/local/gs_datasets/vkitti"
    
    train_dataset, val_dataset = create_mixed_datasets(
        idd_root=idd_root,
        japanese_root=japanese_root,
        vkitti_root=vkitti_root,
        n_frames_input=2,
        n_frames_output=1,
        frame_stride=5,
        target_size=256,
        train_split_ratio=0.8,
        motion_threshold=0.1,
        idd_ratio=0.5,
        japanese_ratio=0.3,
        vkitti_ratio=0.2
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Visualize samples
    visualize_mixed_samples(train_dataset, num_samples=5)
