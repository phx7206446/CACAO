import torch.nn as nn
import torch
import numpy as np
from cross_models.cross_former_model3 import CrossformerV3

class TD(nn.Module):
    def __init__(self, configs, device):
        super(TD, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.timesteps
        self.configs = configs
        self.Wk = nn.ModuleList([nn.Linear(4096, self.num_channels) for i in range(18)])

        self.lsoftmax = nn.LogSoftmax()
        self.device = device


        self.projection_head = nn.Sequential(
            nn.Linear(4096, configs.final_out_channels ),
            nn.BatchNorm1d(configs.final_out_channels ),
            nn.LeakyReLU(inplace=True),
            nn.Linear(configs.final_out_channels , configs.final_out_channels // 4),
        )

        self.crossformer =CrossformerV3(128, 18, 1, 3, win_size=3,
                 factor=10, d_model=32, d_ff=64, n_heads=8, e_layers=3,
                 dropout=0.35, baseline=False, device=device)

    def forward(self, z_aug1, z_aug_comp, for_pred: bool = True):

        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = z_aug_comp.transpose(1, 2)

        batch = z_aug1.shape[0]

        nce = 0

        encode_samples = torch.empty((seq_len, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(0, seq_len):
            encode_samples[i] = z_aug2[:, i, :].view(batch, self.num_channels)

        c_ft = z_aug1[:, :, :]

        _,c_t = self.crossformer(c_ft)


        if for_pred:
            pred = torch.empty((seq_len, batch, self.num_channels)).float().to(self.device)
            for i in np.arange(0, seq_len):
                linear = self.Wk[i]
                pred[i] = linear(c_t)
            for i in np.arange(0, seq_len):
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))

                nce += torch.sum(torch.diag(self.lsoftmax(total)))
            nce /= -1*batch * seq_len

        return nce, self.projection_head(c_t)




