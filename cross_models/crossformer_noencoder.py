import torch
import torch.nn as nn
from einops import repeat
from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.cross_embed import DSW_embedding
from math import ceil
from einops import rearrange, repeat

"""
parser = argparse.ArgumentParser(description='CrossFormer')

parser.add_argument('--data', type=str, required=False, default='HAR', help='data')
parser.add_argument('--epochs', type=int, default=60, help='training_epochs')
parser.add_argument('--data_path', type=str, default=r"E:\MyCode\ChemicalFaber_new\data", help="default_data_path")

parser.add_argument('--data_type', type=str, default="HAR", help="data name options: [chemical_faber, Air, AReM, HAR]")
parser.add_argument('--model_type', type=str, default="Crossformer", help='model name, options: [Crossformer, Autoformer, Informer, Transformer, LSTM, CNN, Crossformer_v2]')
parser.add_argument('--training_mode', type=str, default="self_supervised", help="training mode options: [self_supervised, SupCon, ft, his, avg, jn, train_linear]")
parser.add_argument('--seq_len', type=int, default=128, help="sequence len")
parser.add_argument('--out_len', type=int, default=10, help="out sequence len")
parser.add_argument('--task_mode', type=str, default="classification", help="task mode [classification, regression]")
parser.add_argument('--threshold', type=list, default=[1.2, 1.5], help="for the classification's threshold")

parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--kernel_size', type=int, default=8)
parser.add_argument('--input_channels', type=int,default=9)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--features_len', type=int, default=18)

parser.add_argument('--experiment_description',     default='HAR_experiments_2',  type=str,   help='Experiment Description')
parser.add_argument('--run_description',            default='Crossformer_test1_noencoder_shixuyc', type=str,   help='Experiment Description')
parser.add_argument('--logs_save_dir',              default='experiments_logs',  type=str,   help='saving directory')

parser.add_argument('--final_out_channels', type=int, default=128, help="tc_model_out_feature")
# parser.add_argument('--final_out_channels', type=int, default=62, help="tc_model_out_feature")
# augmentations
parser.add_argument('--jitter_scale_ratio', type=float, default=1.1, help="")
parser.add_argument('--jitter_ratio', type=float, default=0.8, help="")
parser.add_argument('--max_seg', type=int, default=8, help="")
# Context_Cont_configs
parser.add_argument('--temperature', type=float, default=0.2, help="")
parser.add_argument('--use_cosine_similarity', type=bool, default=True, help="")
# TC
parser.add_argument('--hidden_dim', type=int, default=100, help="")
parser.add_argument('--timesteps', type=int, default=6, help="")
parser.add_argument('--split_ratio', type=list, default=[0.6, 0.2, 0.2], help="split_ratio")
parser.add_argument('--hidden_size', type=int, default=100, help="Lstm model hidden_size")
parser.add_argument('--num_classes', type=int, default=7, help="num_classes")
parser.add_argument('--use_bir', type=bool, default=True, help="lstm model's bir")
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')


parser.add_argument('--seg_len', type=int, default=8, help='segment length (L_seg)')
parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
parser.add_argument('--data_dim', type=int, default=9, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--d_model', type=int, default=32, help='dimension of hidden states (d_model)')
# parser.add_argument('--d_model', type=int, default=16, help='dimension of hidden states (d_model)')
parser.add_argument('--merge_win_size', type=int, default=8, help="merge win_size")
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--dropout', type=float, default=0.35, help='dropout')

# learning rate schedule
parser.add_argument('--warm_up_step', type=int, default=10, help="")
parser.add_argument('--max_learning_rate', type=float, default=1e-4, help="")
parser.add_argument('--min_learning_rate', type=float, default=1e-7, help="")
parser.add_argument('--lr_min', type=float, default=1e-7, help="")
parser.add_argument('--lr_type', type=str, default="Cosine_annealing", help="[Cosine_annealing, normal]")
parser.add_argument('--T_0', type=int, default=10)
parser.add_argument('--T_mult', type=int, default=1)

# max_learning_rate: 0.0001
# min_learning_rate: 0.00001
# warm_up_step: 1000
# T_0: 10
# T_mult: 1

parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# model define
parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
parser.add_argument('--c_out', type=int, default=9, help='output size')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--attn_factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')


parser.add_argument('--use_earlystop', type=bool, default=False, help=' use the early stopping')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate_crossformer', type=float, default=3e-4, help='optimizer initial learning rate') ##3e-4
parser.add_argument('--learning_rate', type=float, default=3e-4, help='optimizer initial learning rate') ##3e-4
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--use_adj', type=bool, default=False, help="use adjust learning rate")
parser.add_argument('--lradj', type=str, default='warm_up_cosine', help='adjust learning rate')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--early_stopping_path', type=str, default="E:\MyCode\ChemicalFaber_new\weight\early_stopping_path", help="early stopping save path")
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
"""



class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4,
                 factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                 dropout=0.0, baseline=False, device=torch.device('cuda:0'), num_classes: int = 3):
        super(Crossformer, self).__init__()
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
        # self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1,\
        #                        dropout=0.1, in_seg_num=(self.pad_in_len // seg_len), factor=factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        # self.decoder = Decoder(seg_len, e_layers, d_model, n_heads, d_ff, dropout, \
        #                        out_seg_num=(self.pad_out_len // seg_len), factor=factor, num_classes=num_classes)

        self.conv_block= nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
            nn.Dropout()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
        )

        self.logits = nn.Sequential(
            nn.Linear(8448, 1280),
            nn.Dropout(),
            nn.GELU(),
            nn.Linear(1280, 6)
        )
        self._initialize_weights()

    def forward(self, x_seq):

        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)


        batch_size = x_seq.shape[0]
        dim = x_seq.shape[2]



        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        # enc_out = self.encoder(x_seq)
        # enc_feature_lt = enc_out[-1]
        #


        enc_feature = x_seq.reshape(batch_size, -1, dim)
        # enc_feature = enc_feature_lt.reshape(batch_size, -1, dim)

        enc_feature = enc_feature.permute(0, 2, 1)


        feature = self.conv_block(enc_feature)
        feature = self.conv_block2(feature)
        feature = self.conv_block3(feature)
        # dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        # todo enc_out 作为feature

        # final_predict,final_predict_cs = self.decoder(dec_in, enc_out)

        enc_feature_sc = feature.reshape(batch_size, -1)


        predict_y = self.logits(enc_feature_sc)

        return predict_y, feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)