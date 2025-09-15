#!/usr/bin/env python3
"""
Prediction script for mixed dataset (IDD + Japanese Streets + vKITTI)
Uses the trained checkpoint to perform predictions.
- Always picks 3 frames from the same folder (temporal coherence)
- Ensures different folders are sampled each time for variety
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params
import cv2
from tqdm import tqdm
import json
import glob
import random

# Configuration
CHECKPOINT_PATH = "/home/jovyan/AV_GS4/conv_idd_64/save_model/2025-08-31T06-30-10/checkpoint_28_0.000333.pth.tar"
OUTPUT_DIR = "./prediction_results"
NUM_SAMPLES_PER_DATASET = 20
SAVE_IMAGES = True
SAVE_METRICS = True

# Dataset paths
DATASET_PATHS = {
    'idd': "/mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4",
    'japanese': "/mnt/local/gs_datasets/japanese-streets",
    'vkitti': "/mnt/local/gs_datasets/vkitti"
}

def create_lightweight_dataset():
    """Create datasets where frames are sampled from different folders each time."""
    print("Creating lightweight dataset for prediction...")

    from mixed_dataset import find_idd_sequence_folders, find_japanese_sequence_folders, find_vkitti_sequence_folders

    idd_sequences = find_idd_sequence_folders(DATASET_PATHS['idd'])[:10]
    japanese_sequences = find_japanese_sequence_folders(DATASET_PATHS['japanese'])[:10]
    vkitti_sequences = find_vkitti_sequence_folders(DATASET_PATHS['vkitti'])[:20]

    class LightweightDataset:
        def __init__(self, sequences, dataset_type):
            self.sequences = sequences
            self.dataset_type = dataset_type
            self.length = len(sequences)

        def __getitem__(self, idx):
            try:
                # always pick from one folder but rotate across folders
                seq_idx = idx % len(self.sequences)
                sequence = self.sequences[seq_idx]

                frame_files = sorted(
                    glob.glob(os.path.join(sequence, "*.jpg")) + glob.glob(os.path.join(sequence, "*.jpeg"))
                )

                if len(frame_files) >= 3:
                    chosen_idx = sorted(random.sample(range(len(frame_files)), 3))
                    input_frame1 = cv2.imread(frame_files[chosen_idx[0]])
                    input_frame2 = cv2.imread(frame_files[chosen_idx[1]])
                    output_frame = cv2.imread(frame_files[chosen_idx[2]])

                    # resize
                    input_frame1 = cv2.resize(input_frame1, (256, 256))
                    input_frame2 = cv2.resize(input_frame2, (256, 256))
                    output_frame = cv2.resize(output_frame, (256, 256))

                    # convert to RGB + normalize
                    input_frame1 = cv2.cvtColor(input_frame1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    input_frame2 = cv2.cvtColor(input_frame2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                    # convert to tensors
                    input_frames = torch.from_numpy(np.stack([input_frame1, input_frame2])).permute(0, 3, 1, 2)
                    output_frames = torch.from_numpy(output_frame).unsqueeze(0).permute(0, 3, 1, 2)

                    info = {
                        'dataset': self.dataset_type,
                        'sequence_name': os.path.basename(sequence),
                        'frame_idx': idx
                    }

                    return [idx, output_frames, input_frames, input_frames[-1], info]
                else:
                    raise RuntimeError("Not enough frames in folder")

            except Exception as e:
                print(f"Error loading sample {idx} in {self.dataset_type}: {e}")
                dummy_input = torch.randn(2, 3, 256, 256)
                dummy_output = torch.randn(1, 3, 256, 256)
                info = {'dataset': self.dataset_type, 'frame_idx': idx}
                return [idx, dummy_output, dummy_input, dummy_input[-1], info]

        def __len__(self):
            return self.length * 10  # allow multiple samples per folder

    return (
        LightweightDataset(idd_sequences, 'idd'),
        LightweightDataset(japanese_sequences, 'japanese'),
        LightweightDataset(vkitti_sequences, 'vkitti')
    )

def load_model(checkpoint_path):
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

    encoder = Encoder(*convlstm_encoder_params)
    decoder = Decoder(*convlstm_decoder_params)
    net = ED(encoder, decoder)

    net.load_state_dict(checkpoint['state_dict'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    print(f"Model loaded successfully on {device}")
    return net, device

def calculate_metrics(pred, target):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    mse = np.mean((pred_np - target_np) ** 2)
    psnr = float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
    return mse, psnr

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_prediction_image(inputs, pred, target, sample_info, output_path):
    inputs_np = inputs.cpu().numpy()
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(f'Prediction: {sample_info["dataset"]} - {sample_info["sequence_name"]}', fontsize=14)

    for i in range(inputs_np.shape[1]):
        frame = np.clip(inputs_np[0, i], 0, 1)
        axes[0, i].imshow(frame.transpose(1, 2, 0))
        axes[0, i].set_title(f'Input Frame {i+1}')
        axes[0, i].axis('off')

    pred_frame = np.clip(pred_np[0, 0], 0, 1)
    axes[1, 0].imshow(pred_frame.transpose(1, 2, 0))
    axes[1, 0].set_title('Predicted Frame')
    axes[1, 0].axis('off')

    target_frame = np.clip(target_np[0, 0], 0, 1)
    axes[1, 1].imshow(target_frame.transpose(1, 2, 0))
    axes[1, 1].set_title('Ground Truth Frame')
    axes[1, 1].axis('off')

    axes[2, 0].axis('off')
    axes[2, 1].text(0.5, 0.5, f'MSE: {sample_info["mse"]:.6f}\nPSNR: {sample_info["psnr"]:.2f} dB',
                   ha='center', va='center', transform=axes[2, 1].transAxes, fontsize=12)
    axes[2, 1].set_title('Metrics')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def predict_on_dataset(net, device, dataset, dataset_name, output_dir, num_samples):
    print(f"\n{'='*60}")
    print(f"Predicting on {dataset_name.upper()} dataset")
    print(f"{'='*60}")

    mses, psnrs, predictions = [], [], []
    rng = np.random.default_rng()
    available_indices = list(range(len(dataset)))
    selected_indices = rng.choice(available_indices, min(num_samples, len(available_indices)), replace=False)

    for sample_idx, dataset_idx in enumerate(tqdm(selected_indices, desc=f"Processing {dataset_name}")):
        try:
            idx, targetVar, inputVar, frozen, info = dataset[dataset_idx]
            inputs = inputVar.unsqueeze(0).to(device)
            targets = targetVar.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = net(inputs)

            mse, psnr = calculate_metrics(pred, targets)
            mses.append(mse)
            psnrs.append(psnr)

            pred_info = {
                'dataset': dataset_name,
                'sequence_name': info.get('sequence_name', f'seq_{idx}'),
                'frame_idx': idx,
                'mse': float(mse),
                'psnr': float(psnr) if not np.isinf(psnr) else 100.0
            }
            predictions.append(pred_info)

            if SAVE_IMAGES:
                img_filename = f"{dataset_name}_sample_{sample_idx:03d}.png"
                img_path = os.path.join(output_dir, img_filename)
                save_prediction_image(inputs, pred, targets, pred_info, img_path)

        except Exception as e:
            print(f"Error processing sample {dataset_idx}: {e}")
            continue

    if mses:
        avg_mse = float(np.mean(mses))
        avg_psnr = float(np.mean([p for p in psnrs if not np.isinf(p)]))
        print(f"Processed {len(mses)} samples from {dataset_name}")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")

        return {
            'dataset': dataset_name,
            'num_samples': len(mses),
            'avg_mse': avg_mse,
            'avg_psnr': avg_psnr,
            'predictions': predictions
        }
    else:
        print(f"❌ No samples processed from {dataset_name}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return

    net, device = load_model(CHECKPOINT_PATH)

    try:
        idd_dataset, japanese_dataset, vkitti_dataset = create_lightweight_dataset()
        print("✓ Datasets created successfully")
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return

    all_results = []
    for ds, name in [(idd_dataset, 'idd'), (japanese_dataset, 'japanese'), (vkitti_dataset, 'vkitti')]:
        if len(ds) > 0:
            result = predict_on_dataset(net, device, ds, name, OUTPUT_DIR, NUM_SAMPLES_PER_DATASET)
            if result:
                all_results.append(result)

    if SAVE_METRICS:
        serializable_results = make_json_serializable(all_results)
        with open(os.path.join(OUTPUT_DIR, "prediction_results.json"), 'w') as f:
            json.dump(serializable_results, f, indent=2)

    print("\nPREDICTION COMPLETED")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
