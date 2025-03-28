from torch import nn
from utils import make_layers
import torch


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        # Changed seq_len to 1 since we predict only 1 frame
        inputs, state_stage = rnn(inputs, state, seq_len=1)
        seq_number, batch_size, input_channel, height, width = inputs.size()  # seq_number = 1 now
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,C,H,W (S=1 now)
        return inputs


if __name__ == "__main__":
    from net_params import convlstm_encoder_params, convlstm_decoder_params
    from data import create_idd_datasets  # Use your updated dataset
    from encoder import Encoder
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    encoder = Encoder(convlstm_encoder_params[0],
                      convlstm_encoder_params[1]).cuda()
    decoder = Decoder(convlstm_decoder_params[0],
                      convlstm_decoder_params[1]).cuda()
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    # Use the updated dataset with 2 input frames and 1 output frame
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
        batch_size=8,
        shuffle=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,C,H,W (S=2 now)
        state = encoder(inputs)
        output = decoder(state)
        print(f"Decoder output shape: {output.shape}")  # Should be [B, 1, C, H, W]
        break