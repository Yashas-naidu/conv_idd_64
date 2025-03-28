from torch import nn
from utils import make_layers
import torch
import logging


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index starts from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()  # seq_number = 2 now
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)  # RNN processes 2 frames
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,C,H,W (S=2 now)
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


if __name__ == "__main__":
    from net_params import convgru_encoder_params, convgru_decoder_params
    from data import IDDTemporalDataset, create_idd_datasets  # Use your updated dataset
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    encoder = Encoder(convgru_encoder_params[0],
                      convgru_encoder_params[1]).cuda()

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
        batch_size=4,
        shuffle=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,C,H,W (S=2 now)
        state = encoder(inputs)
        print(f"Encoder output states: {[s[0].shape for s in state]}")  # Debug output
        break