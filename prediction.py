import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data import create_idd_datasets
from data_vkitti import create_vkitti_datasets
from data_japanese import create_japanese_datasets
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convgru_encoder_params as encoder_params, convgru_decoder_params as decoder_params

# Load the model architecture
encoder = Encoder(encoder_params[0], encoder_params[1])
decoder = Decoder(decoder_params[0], decoder_params[1])
net = ED(encoder, decoder)

# Inference configuration
USE_IDD = False
USE_VKITTI = False
USE_JAPANESE = True

# Load the checkpoint (set appropriate checkpoint for the selected dataset)
CHECKPOINT_IDD = "/mnt/local/AV_GS4/conv_idd_64/save_model/idd/2020-03-09T00-00-00/checkpoint_050.pth.tar"
CHECKPOINT_VKITTI = "/mnt/local/AV_GS4/conv_idd_64/save_model/vkitti/2020-03-09T00-00-00/checkpoint_050.pth.tar"
CHECKPOINT_JAPANESE = "/mnt/local/AV_GS4/conv_idd_64/save_model/japanese-streets/2020-03-09T00-00-00/checkpoint_050.pth.tar"

if sum([USE_IDD, USE_VKITTI, USE_JAPANESE]) != 1:
    raise ValueError("Exactly one of USE_IDD/USE_VKITTI/USE_JAPANESE must be True for prediction.")

checkpoint_path = CHECKPOINT_JAPANESE if USE_JAPANESE else (CHECKPOINT_VKITTI if USE_VKITTI else CHECKPOINT_IDD)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

# Load the model state dict
net.load_state_dict(checkpoint['state_dict'])

# Move the model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Set the model to evaluation mode
net.eval()

# Prepare the data
IDD_PATH = "/mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4"
VKITTI_PATH = "/mnt/local/gs_datasets/vkitti"
JAPANESE_PATH = "/mnt/local/gs_datasets/japanese-streets"

common_kwargs = dict(
    n_frames_input=2,
    n_frames_output=1,
    frame_stride=5,
    target_size=512,
    train_split_ratio=0.8,
    seed=None,
)

if USE_IDD:
    _, val_dataset = create_idd_datasets(dataset_root=IDD_PATH, **common_kwargs)
elif USE_VKITTI:
    _, val_dataset = create_vkitti_datasets(dataset_root=VKITTI_PATH, **common_kwargs)
else:
    _, val_dataset = create_japanese_datasets(dataset_root=JAPANESE_PATH, **common_kwargs)

validLoader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Output directory for saving predictions
pred_dir_name = 'idd-predictions' if USE_IDD else ('vkitti-predictions' if USE_VKITTI else 'japanese-predictions')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), pred_dir_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters for limiting predictions
SAVE_LIMIT = 10  # Total number of predictions to save
num_batches_to_process = 1000  # Upper bound; we will stop at SAVE_LIMIT
num_samples_per_batch = 64     # Upper bound per batch; we will stop at SAVE_LIMIT

# Make predictions and display input vs predicted vs ground truth frames
total_saved = 0
with torch.no_grad():
    for batch_idx, (idx, targetVar, inputVar, frozen, info) in enumerate(validLoader):
        if batch_idx >= num_batches_to_process:
            break  # Stop after processing the specified number of batches

        inputs = inputVar.to(device)  # Shape: [B, 2, 3, 512, 512]
        targets = targetVar.to(device)  # Shape: [B, 1, 3, 512, 512]

        # Predict the next frame
        pred = net(inputs)  # Shape: [B, 1, 3, 512, 512]

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

            # Save predicted frame and composite visualization
            # Build a descriptive filename when sequence info is available
            seq_name = None
            if isinstance(info, list) and len(info) > sample_idx and isinstance(info[sample_idx], dict):
                meta = info[sample_idx]
                seq_name = f"{meta.get('sequence_name', 'seq')}_idx{meta.get('frame_idx', 0)}"
            fname_base = seq_name if seq_name else f"batch{batch_idx}_sample{sample_idx}"
            pred_img_path = os.path.join(OUTPUT_DIR, f"{fname_base}_pred.png")
            plt.imsave(pred_img_path, frame.transpose(1, 2, 0))

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
            composite_path = os.path.join(OUTPUT_DIR, f"{fname_base}_viz.png")
            plt.savefig(composite_path)
            plt.close()

            total_saved += 1
            if total_saved >= SAVE_LIMIT:
                break
        if total_saved >= SAVE_LIMIT:
            break
