from torch import nn
import torch.nn.functional as F
import torch
from utils import make_layers


class activation():
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class ED(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)  # Input: [B, 2, C, H, W]
        output = self.decoder(state)  # Output: [B, 1, C, H, W]
        return output


if __name__ == "__main__":
    from encoder import Encoder
    from decoder import Decoder
    from net_params import convgru_encoder_params, convgru_decoder_params
    from data import create_idd_datasets
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    encoder = Encoder(convgru_encoder_params[0], convgru_encoder_params[1]).cuda()
    decoder = Decoder(convgru_decoder_params[0], convgru_decoder_params[1]).cuda()
    model = ED(encoder, decoder).cuda()

    # Use the updated dataset
    dataset_root = r"C:\Users\YASHAS\capstone\baselines\conv_idd_64\idd_temporal_train_4"
    train_dataset, _ = create_idd_datasets(
        dataset_root=dataset_root,
        n_frames_input=2,
        n_frames_output=1,
        frame_stride=5,
        target_size=256,
        train_split_ratio=0.8
    )

    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,C,H,W (S=2)
        output = model(inputs)  # B,S,C,H,W (S=1)
        print(f"Model output shape: {output.shape}")  # Should be [B, 1, C, H, W]
        break