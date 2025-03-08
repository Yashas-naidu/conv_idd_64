import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import random

# Step 1: Extract all frames from the video (60 frames per second)
def extract_frames(video_path, output_folder, train_split_ratio=0.8):
    """Extract all frames from a video and save them to train and validation folders."""
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'validation')
    
    # Create output folders
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    duration = frame_count / fps  # Total duration in seconds
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    saved_frame_count = 0
    train_count = 0
    val_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Randomly split into train and validation sets
        if random.random() < train_split_ratio:
            frame_filename = os.path.join(train_folder, f"frame_{train_count:04d}.jpg")
            train_count += 1
        else:
            frame_filename = os.path.join(val_folder, f"frame_{val_count:04d}.jpg")
            val_count += 1
        
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1
    
    cap.release()
    print(f"Finished extracting {saved_frame_count} frames")
    print(f"Train frames: {train_count}")
    print(f"Validation frames: {val_count}")

# Step 2: Dataset class for RGB car videos
class MovingCarsRGB(data.Dataset):
    def __init__(self, frames_folder, n_frames_input, n_frames_output, 
                 target_size=256, transform=None):
        '''
        Dataset for high-resolution car videos with colored (RGB) frames
        
        Args:
            frames_folder: path to the folder containing extracted frames
            n_frames_input: number of input frames
            n_frames_output: number of output frames to predict
            target_size: size to resize frames to (square dimensions for the model)
            transform: optional additional transformations
        '''
        super(MovingCarsRGB, self).__init__()
        
        self.frames_folder = frames_folder
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.target_size = target_size
        self.transform = transform
        
        # Load RGB video frames
        self.frames = self.load_frames_from_folder()
        self.length = max(1, len(self.frames) - self.n_frames_total + 1)
    
    def load_frames_from_folder(self):
        """Load and preprocess frames from the folder"""
        frames = []
        frame_files = sorted(os.listdir(self.frames_folder))
        
        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Process the high-resolution RGB frame
            processed_frame = self.preprocess_frame(frame)
            frames.append(processed_frame)
        
        print(f"Loaded {len(frames)} frames from {self.frames_folder}")
        return frames
    
    def preprocess_frame(self, frame):
        """Process a high-resolution frame in RGB"""
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
        
        # Get sequence of frames
        sequence = self.frames[idx:idx + self.n_frames_total]
        sequence = np.array(sequence)
        
        # Split into input and output
        input_frames = sequence[:self.n_frames_input]
        
        if self.n_frames_output > 0:
            output_frames = sequence[self.n_frames_input:self.n_frames_total]
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
        
        # Format to match the original MovingMNIST structure
        # [idx, output_frames, input_frames, frozen, _]
        return [idx, output_tensor, input_tensor, frozen, np.zeros(1)]
    
    def __len__(self):
        return self.length

# Step 3: Create the dataset and DataLoader
def create_rgb_car_dataset(
    video_path, 
    n_frames_input=10, 
    n_frames_output=10,
    target_size=64,
    train_split_ratio=0.8
):
    """Create a colored (RGB) dataset from a car video file"""
    # Extract frames first (all frames)
    output_folder = os.path.join(os.path.dirname(video_path), "extracted_frames")
    extract_frames(video_path, output_folder, train_split_ratio)
    
    # Create train and validation datasets
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'validation')
    
    train_dataset = MovingCarsRGB(
        frames_folder=train_folder,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        target_size=target_size
    )
    
    val_dataset = MovingCarsRGB(
        frames_folder=val_folder,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        target_size=target_size
    )
    
    return train_dataset, val_dataset

# Main execution block
if __name__ == '__main__':
    # Example usage
    video_path = r'C:\Users\YASHAS\capstone\baselines\conv_idd_64\Cars.mp4'  # Replace with your actual video path

    # Create the datasets
    train_dataset, val_dataset = create_rgb_car_dataset(
        video_path=video_path,
        n_frames_input=10,
        n_frames_output=10,
        target_size=64,
        train_split_ratio=0.8
    )

    # Create DataLoaders for train and validation
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,           # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4           # For parallel loading
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,           # Adjust based on your GPU memory
        shuffle=False,
        num_workers=4           # For parallel loading
    )

