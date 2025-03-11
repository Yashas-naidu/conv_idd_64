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

# Load the checkpoint
checkpoint_path = r'C:\Users\YASHAS\capstone\baselines\conv_idd_64\save_model\2020-03-09T00-00-00\checkpoint_0_0.004469.pth.tar'
checkpoint = torch.load(checkpoint_path)

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
    n_frames_input=10,
    n_frames_output=10,
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

        inputs = inputVar.to(device)
        targets = targetVar.to(device)

        # Predict the next frames
        pred = net(inputs)

        # Convert to NumPy
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        pred = pred.cpu().numpy()

        # Iterate over each sample in the batch
        for sample_idx in range(min(num_samples_per_batch, pred.shape[0])):
            # Create a figure to display input, predicted, and ground truth frames
            plt.figure(figsize=(20, 9))

            # Display the input frames (Row 1)
            for i in range(inputs.shape[1]):  # Loop over input frames
                plt.subplot(3, inputs.shape[1], i + 1)  # 3 rows, 10 columns
                frame = inputs[sample_idx, i, :, :, :]  # RGB frame (3 channels)
                frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
                plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C) for visualization
                plt.title(f'Input {i + 1}')
                plt.axis('off')

            # Display the reconstructed frames (Row 2 - Model Predictions)
            for i in range(pred.shape[1]):  # Loop over predicted frames
                plt.subplot(3, pred.shape[1], inputs.shape[1] + i + 1)
                frame = pred[sample_idx, i, :, :, :]  # RGB frame (3 channels)
                frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
                plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C) for visualization
                plt.title(f'Predicted {i + 1}')
                plt.axis('off')

            # Display the ground truth frames (Row 3)
            for i in range(targets.shape[1]):  # Loop over ground truth frames
                plt.subplot(3, targets.shape[1], 2 * inputs.shape[1] + i + 1)
                frame = targets[sample_idx, i, :, :, :]  # RGB frame (3 channels)
                frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
                plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C) for visualization
                plt.title(f'Ground Truth {i + 1}')
                plt.axis('off')

            plt.suptitle(f'Batch {batch_idx}, Sample {sample_idx}\n'
                         f'Top: Input Frames | Middle: Predicted Frames | Bottom: Ground Truth Frames',
                         fontsize=16)
            plt.tight_layout()
            plt.show()