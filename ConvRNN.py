import torch
import torch.nn as nn


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=2):
        # seq_len=2 for encoder (2 input frames), seq_len=1 for decoder (1 output frame)
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=2):
        # seq_len=2 for encoder (2 input frames), seq_len=1 for decoder (1 output frame)
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)


if __name__ == "__main__":
    from data import create_idd_datasets
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Test CGRU_cell
    cgru = CGRU_cell(shape=(256, 256), input_channels=3, filter_size=3, num_features=64).cuda()
    dataset_root = r"C:\Users\YASHAS\capstone\baselines\conv_idd_64\idd_temporal_train_4"
    train_dataset, _ = create_idd_datasets(
        dataset_root=dataset_root,
        n_frames_input=2,
        n_frames_output=1,
        frame_stride=5,
        target_size=256,
        train_split_ratio=0.8
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
    for i, (idx, targetVar, inputVar, _, _) in enumerate(train_loader):
        inputs = inputVar.cuda()  # [B, 2, 3, 256, 256]
        output, hidden = cgru(inputs, None, seq_len=2)
        print(f"CGRU output shape: {output.shape}, hidden shape: {hidden.shape}")
        break

    # Test CLSTM_cell
    clstm = CLSTM_cell(shape=(256, 256), input_channels=3, filter_size=3, num_features=64).cuda()
    for i, (idx, targetVar, inputVar, _, _) in enumerate(train_loader):
        inputs = inputVar.cuda()  # [B, 2, 3, 256, 256]
        output, (hx, cx) = clstm(inputs, None, seq_len=2)
        print(f"CLSTM output shape: {output.shape}, hx shape: {hx.shape}, cx shape: {cx.shape}")
        break