import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data import create_idd_datasets 
from data_vkitti import create_vkitti_datasets
from data_japanese import create_japanese_datasets
from losses import SSIMLoss
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
IDD_PATH = "/mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4"
VKITTI_PATH = "/mnt/local/gs_datasets/vkitti"
JAPANESE_PATH = "/mnt/local/gs_datasets/japanese-streets"

# Hardcoded training/config parameters
USE_IDD = False       # Toggle dataset
USE_VKITTI = False     # Toggle dataset
USE_JAPANESE = True  # Toggle dataset
CONVLSTM = True
CONVGRU = False
BATCH_SIZE = 2
LR = 1e-4
FRAMES_INPUT = 2
FRAMES_OUTPUT = 1
EPOCHS = 0
MOTION_THRESHOLD = 0.1

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

def train():
    '''
    main function to run the training
    '''
    # Ensure exactly one dataset is selected
    if sum([USE_IDD, USE_VKITTI, USE_JAPANESE]) != 1:
        raise ValueError("Exactly one of USE_IDD, USE_VKITTI, USE_JAPANESE must be True.")

    # Create train and validation datasets
    if USE_VKITTI:
        train_dataset, val_dataset = create_vkitti_datasets(
            dataset_root=VKITTI_PATH,
            n_frames_input=FRAMES_INPUT,
            n_frames_output=FRAMES_OUTPUT,
            frame_stride=5,
            target_size=512,
            train_split_ratio=0.8,
            seed=random_seed,
            motion_threshold=MOTION_THRESHOLD,
        )
    elif USE_JAPANESE:
        train_dataset, val_dataset = create_japanese_datasets(
            dataset_root=JAPANESE_PATH,
            n_frames_input=FRAMES_INPUT,
            n_frames_output=FRAMES_OUTPUT,
            frame_stride=5,
            target_size=512,
            train_split_ratio=0.8,
            seed=random_seed,
            motion_threshold=MOTION_THRESHOLD,
        )
    elif USE_IDD:
        train_dataset, val_dataset = create_idd_datasets(
            dataset_root=IDD_PATH,
            n_frames_input=FRAMES_INPUT,
            n_frames_output=FRAMES_OUTPUT,
            frame_stride=5,
            target_size=512,
            train_split_ratio=0.8,
            seed=random_seed,
            motion_threshold=MOTION_THRESHOLD,
        )
    else:
        raise RuntimeError("Dataset toggle configuration invalid.")

    # Create DataLoaders
    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=20,  # Temporarily set to 0 to debug
        pin_memory=True
    )

    validLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=20,  # Temporarily set to 0 to debug
        pin_memory=True
    )

    if CONVLSTM:
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params
    if CONVGRU:
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params
    else:
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params

    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = SSIMLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                    )

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, EPOCHS + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W (S=2 now)
            label = targetVar.to(device)  # B,S,C,H,W (S=1 now)
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W (S=1 now)
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / BATCH_SIZE
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
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / BATCH_SIZE
                # record validation loss
                valid_losses.append(loss_aver)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(EPOCHS))

        print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()