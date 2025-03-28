import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data import create_idd_datasets  # Updated import
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convgru_encoder_params as encoder_params, convgru_decoder_params as decoder_params

# Load the model architecture
encoder = Encoder(encoder_params[0], encoder_params[1])
decoder = Decoder(decoder_params[0], decoder_params[1])
net = ED(encoder, decoder)

# Load the checkpoint (update this path to your new checkpoint if retrained)
checkpoint_path = r'C:\Users\YASHAS\capstone\conv_idd_64\save_model\2020-03-09T00-00-00\checkpoint_0_0.022345.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

# Load the model state dict
net.load_state_dict(checkpoint['state_dict'])

# Move the model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Set the model to evaluation mode
net.eval()

# Prepare the data
dataset_root = r'C:\Users\YASHAS\capstone\baselines\conv_idd_64\idd_temporal_train_4'  # Path to the dataset root folder
_, val_dataset = create_idd_datasets(
    dataset_root=dataset_root,
    n_frames_input=2,         # Updated to 2 input frames
    n_frames_output=1,        # Updated to 1 output frame
    frame_stride=5,           # Added stride of 5
    target_size=256,
    train_split_ratio=0.8,
    seed=None
)

validLoader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Parameters for limiting predictions
num_batches_to_process = 1  # Number of batches to process
num_samples_per_batch = 1   # Number of samples per batch to process

# Make predictions and display input vs predicted vs ground truth frames
with torch.no_grad():
    for batch_idx, (idx, targetVar, inputVar, frozen, _) in enumerate(validLoader):
        if batch_idx >= num_batches_to_process:
            break  # Stop after processing the specified number of batches

        inputs = inputVar.to(device)  # Shape: [B, 2, 3, 256, 256]
        targets = targetVar.to(device)  # Shape: [B, 1, 3, 256, 256]

        # Predict the next frame
        pred = net(inputs)  # Shape: [B, 1, 3, 256, 256]

        # Convert to NumPy
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        pred = pred.cpu().numpy()

        # Iterate over each sample in the batch
        for sample_idx in range(min(num_samples_per_batch, pred.shape[0])):
            # Create a figure to display input, predicted, and ground truth frames
            plt.figure(figsize=(12, 6))  # Adjusted size for 2 inputs + 1 output

            # Display the input frames (Row 1)
            for i in range(inputs.shape[1]):  # Loop over 2 input frames
                plt.subplot(3, 2, i + 1)  # 3 rows, 2 columns
                frame = inputs[sample_idx, i, :, :, :]  # RGB frame (3 channels)
                frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
                plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
                plt.title(f'Input Frame {i * 5 + 1}')  # Frame 1, Frame 6 (stride 5)
                plt.axis('off')

            # Display the predicted frame (Row 2)
            plt.subplot(3, 2, 3)  # Middle row, first column
            frame = pred[sample_idx, 0, :, :, :]  # Single predicted frame
            frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Predicted Frame 11')  # Frame 11 (5 frames after second input)
            plt.axis('off')

            # Display the ground truth frame (Row 3)
            plt.subplot(3, 2, 5)  # Bottom row, first column
            frame = targets[sample_idx, 0, :, :, :]  # Single ground truth frame
            frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Ground Truth Frame 11')  # Frame 11
            plt.axis('off')

            plt.suptitle(f'Batch {batch_idx}, Sample {sample_idx}\n'
                         f'Top: Input Frames (1, 6) | Middle: Predicted Frame (11) | Bottom: Ground Truth Frame (11)',
                         fontsize=12)
            plt.tight_layout()
            plt.show()
