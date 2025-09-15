#!/usr/bin/env python3
"""
Training script for mixed dataset (IDD + Japanese Streets + vKITTI)
Usage: python train_mixed.py
(All configuration is hardcoded in the script)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from mixed_dataset import create_mixed_datasets
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import datetime
import math

# Hardcoded training parameters
TRAINING_CONFIG = {
    'batch_size': 45,
    'learning_rate': 1e-4,
    'frames_input': 2,
    'frames_output': 1,
    'epochs': 50,
    'motion_threshold': 0.1,
    'seed': 1996
}

# Hardcoded dataset paths
DATASET_PATHS = {
    'idd': "/mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4",
    'japanese': "/mnt/local/gs_datasets/japanese-streets",
    'vkitti': "/mnt/local/gs_datasets/vkitti"
}

# Dataset ratios
DATASET_RATIOS = {
    'idd': 0.5,      # 50% IDD
    'japanese': 0.3, # 30% Japanese
    'vkitti': 0.2    # 20% vKITTI
}

# Create timestamp for this training run
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# Set random seeds for reproducibility
random_seed = TRAINING_CONFIG['seed']
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create save directory
save_dir = f'./save_model/{TIMESTAMP}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def calculate_psnr(mse):
    """Calculate PSNR from MSE"""
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def train():
    '''
    Main function to run the training
    '''
    print("=" * 60)
    print("MIXED DATASET TRAINING - HARDCODED CONFIGURATION")
    print("=" * 60)
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"Motion Threshold: {TRAINING_CONFIG['motion_threshold']}")
    print(f"Frames Input: {TRAINING_CONFIG['frames_input']}")
    print(f"Frames Output: {TRAINING_CONFIG['frames_output']}")
    print("=" * 60)
    print(f"Starting training with mixed dataset...")
    
    # Use hardcoded dataset paths and ratios
    dataset_paths = DATASET_PATHS
    dataset_ratios = DATASET_RATIOS
    
    print(f"IDD: {dataset_ratios['idd']*100}% from {dataset_paths['idd']}")
    print(f"Japanese: {dataset_ratios['japanese']*100}% from {dataset_paths['japanese']}")
    print(f"vKITTI: {dataset_ratios['vkitti']*100}% from {dataset_paths['vkitti']}")
    
    # Create train and validation datasets using MixedTemporalDataset
    train_dataset, val_dataset = create_mixed_datasets(
        idd_root=dataset_paths['idd'],
        japanese_root=dataset_paths['japanese'],
        vkitti_root=dataset_paths['vkitti'],
        n_frames_input=TRAINING_CONFIG['frames_input'],
        n_frames_output=TRAINING_CONFIG['frames_output'],
        frame_stride=5,
        target_size=256,
        train_split_ratio=0.8,
        seed=random_seed,
        motion_threshold=TRAINING_CONFIG['motion_threshold'],
        idd_ratio=dataset_ratios['idd'],
        japanese_ratio=dataset_ratios['japanese'],
        vkitti_ratio=dataset_ratios['vkitti']
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DataLoaders
    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=40,
        pin_memory=True
    )

    validLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=40,
        pin_memory=True
    )

    # Select model architecture (using ConvLSTM)
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
    model_type = "ConvLSTM"

    print(f"Using {model_type} architecture")

    # Create model
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)

    # Setup TensorBoard
    run_dir = f'./runs/{TIMESTAMP}'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    
    net.to(device)

    # Load existing model if available
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        print('==> Loading existing model')
        model_info = torch.load(checkpoint_path)
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
        print(f'==> Resuming from epoch {cur_epoch}')
    else:
        cur_epoch = 0
        print('==> Starting training from scratch')

    # Setup loss function and optimizer
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4
    )

    # Training tracking variables
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    # Metrics tracking
    train_mses = []
    valid_mses = []
    train_psnrs = []
    valid_psnrs = []

    print(f"Starting training for {TRAINING_CONFIG['epochs']} epochs...")
    
    for epoch in range(cur_epoch, TRAINING_CONFIG['epochs'] + 1):
        ###################
        # Train the model #
        ###################
        net.train()
        t = tqdm(trainLoader, leave=False, total=len(trainLoader), desc=f'Epoch {epoch}/{TRAINING_CONFIG["epochs"]}')
        
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W (S=2)
            label = targetVar.to(device)  # B,S,C,H,W (S=1)
            
            optimizer.zero_grad()
            pred = net(inputs)  # B,S,C,H,W (S=1)
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / TRAINING_CONFIG['batch_size']
            train_losses.append(loss_aver)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        
        tb.add_scalar('TrainLoss', loss_aver, epoch)

        ######################
        # Validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader), desc='Validation')
            
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:  # Limit validation to 3000 batches
                    break
                    
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / TRAINING_CONFIG['batch_size']
                valid_losses.append(loss_aver)
                
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()

        # Calculate average losses and metrics
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        # Calculate PSNR from MSE
        train_psnr = calculate_psnr(train_loss)
        valid_psnr = calculate_psnr(valid_loss)
        train_psnrs.append(train_psnr)
        valid_psnrs.append(valid_psnr)
        
        # Store MSE values
        train_mses.append(train_loss)
        valid_mses.append(valid_loss)
        
        # Log metrics to TensorBoard
        tb.add_scalar('TrainMSE', train_loss, epoch)
        tb.add_scalar('ValidMSE', valid_loss, epoch)
        tb.add_scalar('TrainPSNR', train_psnr, epoch)
        tb.add_scalar('ValidPSNR', valid_psnr, epoch)

        # Print epoch statistics
        epoch_len = len(str(TRAINING_CONFIG['epochs']))
        print_msg = (f'[{epoch:>{epoch_len}}/{TRAINING_CONFIG['epochs']:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f} ' +
                     f'train_psnr: {train_psnr:.2f} ' +
                     f'valid_psnr: {valid_psnr:.2f}')
        print(print_msg)

        # Clear lists for next epoch
        train_losses = []
        valid_losses = []
        
        # Update learning rate
        pla_lr_scheduler.step(valid_loss)

        # Save checkpoint
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': TRAINING_CONFIG,
            'dataset_paths': DATASET_PATHS,
            'dataset_ratios': DATASET_RATIOS
        }
        
        early_stopping(valid_loss, model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save final loss and metrics history
    with open(os.path.join(save_dir, "avg_train_losses.txt"), 'wt') as f:
        for loss in avg_train_losses:
            print(loss, file=f)

    with open(os.path.join(save_dir, "avg_valid_losses.txt"), 'wt') as f:
        for loss in avg_valid_losses:
            print(loss, file=f)
    
    # Save MSE metrics
    with open(os.path.join(save_dir, "train_mses.txt"), 'wt') as f:
        for mse in train_mses:
            print(mse, file=f)
    
    with open(os.path.join(save_dir, "valid_mses.txt"), 'wt') as f:
        for mse in valid_mses:
            print(mse, file=f)
    
    # Save PSNR metrics
    with open(os.path.join(save_dir, "train_psnrs.txt"), 'wt') as f:
        for psnr in train_psnrs:
            print(psnr, file=f)
    
    with open(os.path.join(save_dir, "valid_psnrs.txt"), 'wt') as f:
        for psnr in valid_psnrs:
            print(psnr, file=f)

    # Print final metrics summary
    print(f"\n" + "="*60)
    print("FINAL TRAINING METRICS")
    print("="*60)
    print(f"Final Train MSE: {train_mses[-1]:.6f}")
    print(f"Final Valid MSE: {valid_mses[-1]:.6f}")
    print(f"Final Train PSNR: {train_psnrs[-1]:.2f} dB")
    print(f"Final Valid PSNR: {valid_psnrs[-1]:.2f} dB")
    print(f"Best Valid PSNR: {max(valid_psnrs):.2f} dB")
    print(f"Best Valid MSE: {min(valid_mses):.6f}")
    print("="*60)
    
    print(f"Training completed. Model saved to {save_dir}")
    print(f"TensorBoard logs saved to {run_dir}")
    print(f"Metrics saved to text files in {save_dir}")

if __name__ == "__main__":
    train()
