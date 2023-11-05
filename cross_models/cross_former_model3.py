import torch
import torch.nn as nn
from einops import repeat
from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.cross_embed import DSW_embedding
from math import ceil
from einops import rearrange, repeat


class CrossformerV3(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4,
                 factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                 dropout=0.0, baseline=False, device=torch.device('cuda:0'), num_classes: int = 3):
        super(CrossformerV3, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1,\
                               dropout=0.5, in_seg_num=(self.pad_in_len // seg_len), factor=factor)

        # self.conv_block= nn.Sequential(
        #     nn.Conv1d(9, 32, kernel_size=2, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(32),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
        #     # nn.Dropout(0.2)
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv1d(32, 64, kernel_size=2, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(64),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
        #     # nn.Dropout(0.2)
        # )
        #
        # self.conv_block3 = nn.Sequential(
        #     nn.Conv1d(64, 128, kernel_size=2, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(128),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
        #     # nn.Dropout(0.2)
        # )

## e_layer 2
        # self.logits = nn.Sequential(
        #     nn.Linear(2304, 1280),
        #     nn.Dropout(0.2),
        #     nn.GELU(),
        #     nn.Linear(1280, 6),
        # )
        # self._initialize_weights()

    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        batch_size = x_seq.shape[0]
        dim = x_seq.shape[2]

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)


        # enc_feature_lt = enc_out[-1]
        # #
        #
        # # enc_feature = x_seq.reshape(batch_size, -1, dim)
        # enc_feature = enc_feature_lt.reshape(batch_size, -1, dim)
        #
        # enc_feature = enc_feature.permute(0, 2, 1)

        # dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        # predict_y = self.decoder(dec_in, enc_out)

        # predict_y = predict_y.permute(0, 2, 1)

        # feature = self.conv_block(predict_y)
        # feature = self.conv_block2(feature)
        # feature = self.conv_block3(feature)
        # dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        # todo enc_out 作为feature

        # final_predict,final_predict_cs = self.decoder(dec_in, enc_out)

        enc_feature_sc = enc_out[-1].reshape(batch_size, -1)
        #
        #
        # predict_y = self.logits(enc_feature_sc)

        return 1, enc_feature_sc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)