import sys
import torch
from tensorboardX import SummaryWriter
import argparse
from cross_models.data_process import load_data
from torch import nn as nn
from model.TDModel import TD
from utils import EarlyStopping, model_selection, adjust_learning_rate_sche, train_epoch, val_epoch, test_epoch, \
    load_pretrained_weight, log_setting, gen_pseudo_labels, build_optimizer, Model_evaluation
import os
from utils import feature_show_distance


parser = argparse.ArgumentParser(description='CACAO')

parser.add_argument('--data', type=str, required=False, default='HAR', help='data')
parser.add_argument('--epochs', type=int, default=200, help='training_epochs')
parser.add_argument('--data_path', type=str, default=r"data", help="default_data_path")

parser.add_argument('--data_type', type=str, default="HAR", help="data name options: [chemical_faber, Air, AReM, HAR]")
parser.add_argument('--task_type',type=str, default="classification", help='for the task type optional [classification, prediction')
parser.add_argument('--model_type', type=str, default="CNN", help='model name, options: [Crossformer, Autoformer, Informer, Transformer, LSTM, CNN, Crossformer_v2]')
parser.add_argument('--training_mode', type=str, default="self_supervised", help="training mode options: [self_supervised, SupCon, ft, his, avg, jn, train_linear, gen_pseudo_labels]")
parser.add_argument('--seq_len', type=int, default=128, help="sequence len")

parser.add_argument("--use_td", type=bool, default=True)
parser.add_argument('--task_mode', type=str, default="classification", help="task mode [classification, regression]")
parser.add_argument('--threshold', type=list, default=[1.2, 1.5], help="for the classification's threshold")

parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--kernel_size', type=int, default=8)
parser.add_argument('--input_channels', type=int,default=9)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--features_len', type=int, default=18)
parser.add_argument('--experiment_description',     default='HAR_experiment',  type=str,   help='hkanzExperiment Description')
parser.add_argument('--run_description',            default='har_self_supervised_experiment', type=str,   help='Experiment Description')
parser.add_argument('--logs_save_dir',              default='experiments_logs',  type=str,   help='saving directory')

parser.add_argument('--final_out_channels', type=int, default=128, help="tc_model_out_feature")
# augmentations
parser.add_argument('--jitter_scale_ratio', type=float, default=1.1, help="")
parser.add_argument('--jitter_ratio', type=float, default=0.8, help="")
parser.add_argument('--max_seg', type=int, default=32, help="")
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

parser.add_argument("--data_aug_mode", type=str, default="normal", help="options: [permute, comb, scaling, shift, overturn] ")

parser.add_argument("--use_mask", type=bool, default=False)
parser.add_argument("--mask_mode", type=str, default="3dmask", help="options: [dim, time]")
parser.add_argument("--mask_type", type=str, default="binomial", help="options: [binomial, continuous, all_true, all_false, mask_last]")

## comb_parameter
parser.add_argument('--comb_sigma', type=float, default=0.8)
parser.add_argument('--comb_replace_num', type=int, default=5)
parser.add_argument('--comb_min_ts', type=int, default=4)
parser.add_argument('--comb_max_ts', type=int, default=8)
parser.add_argument('--comb_use_per', type=bool, default=True)
parser.add_argument('--comb_use_rs', type=bool, default=False)
parser.add_argument('--comb_use_noise', type=bool, default=False)
## overturn parameter
parser.add_argument('--overturn_max_segments', type=int, default=8)
parser.add_argument('--overturn_ratio', type=float, default=0.6)
parser.add_argument('--overturn_use_partial', type=bool, default=True)
parser.add_argument('--overturn_use_noise', type=bool, default=True)
parser.add_argument('--overturn_sigma', type=float, default=0.8)
# parser.add_argument('--overturn_sigma', type=float, default=0.001)

## shift parameter
parser.add_argument('--shift_max_crop_len', type=int, default=20)
parser.add_argument('--shift_use_scaling', type=bool, default=False)
parser.add_argument('--shift_sigma', type=float, default=1.1)


parser.add_argument('--logistic_dropout', type=float, default=0.8, help='logistic dropout_ratio')
parser.add_argument('--seg_len', type=int, default=3, help='segment length (L_seg)')
parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
parser.add_argument('--data_dim', type=int, default=9, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--merge_win_size', type=int, default=4, help="merge win_size")
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

# learning rate schedule
parser.add_argument('--warm_up_step', type=int, default=10, help="")
parser.add_argument('--max_learning_rate', type=float, default=3e-4, help="")
parser.add_argument('--min_learning_rate', type=float, default=1e-7, help="")
parser.add_argument('--lr_min', type=float, default=1e-7, help="")
parser.add_argument('--lr_type', type=str, default="Cosine_annealing", help="[Cosine_annealing, normal]")
parser.add_argument('--T_0', type=int, default=10)
parser.add_argument('--T_mult', type=int, default=1)

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
parser.add_argument('--lambda1', type=float, default=1, help='lambda1')
parser.add_argument('--lambda2', type=float, default=0.7, help='lambda1')
## todozzzz
parser.add_argument('--loss_type', type=str, default='tc', help="C loss  optional [tc,c,mc,mc_tc,mask]")
parser.add_argument('--m_lambda', type=list, default=[], help='Multi-loss lambda')


parser.add_argument('--use_earlystop', type=bool, default=False, help=' use the early stopping')
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--learning_rate_crossformer', type=float, default=3e-4, help='optimizer initial learning rate') ##3e-4
parser.add_argument('--learning_rate', type=float, default=3e-4, help='optimizer initial learning rate') ##3e-4
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--use_adj', type=bool, default=False, help="use adjust learning rate")
parser.add_argument('--lradj', type=str, default='linear_decay', help='adjust learning rate  choosen option [warm_up_cosine, weight_decay, Cosine_annealing, linear_decay]')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--early_stopping_path', type=str, default="E:\MyCode\ChemicalFaber_new\weight\early_stopping_path", help="early stopping save path")
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

args = parser.parse_args()

# args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
# args.use_gpu

"""
todo 1.Informer方式encoder,Reformer,Transformer
2.将encoder中的time enc 和 dim enc 输出做对比
3. cnn encoder 预测任务 + time_enc做对比
"""
if args.use_gpu:
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    device = torch.device('cuda:{}'.format(args.gpu))
    print('Use GPU: cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')
    print('Use CPU')


if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)


def run_tesvec(args):
    logger, experiment_log_dir = log_setting(args)
    # load data
    loader, data_size, _ = load_data(args.data_type, args.data_path, args.seq_len, args.training_mode, args.batch_size,
                                     args)
    logger.debug("Data loaded ...")
    logger.debug("original  c ")
    train_loader, val_loader, test_loader = loader
    model_evaluate = Model_evaluation(args, experiment_log_dir)
    criterion = nn.CrossEntropyLoss()



def runs(args):
    # setting the seed and logger
    logger, experiment_log_dir = log_setting(args)

    # load data
    if args.task_type == "classification":
        loader, data_size, _ = load_data(args.data_type, args.data_path, args.seq_len, args.training_mode,
                                         args.batch_size, args)
        train_loader, val_loader, test_loader = loader
        criterion = nn.CrossEntropyLoss()

    logger.debug("Data loaded ...")
    logger.debug("original  c ")

    model_evaluate = Model_evaluation(args, experiment_log_dir)
    # build model
    base_model = model_selection(args)
    # build loss_function

    # build base_model optimizer
    tb_writer = SummaryWriter("run")
    base_model.to(device)
    if args.use_td:
        tc_model = TD(args, device).to(device)
        # tc_model = TD2(args, device).to(device)
    else:
        tc_model = TC(args, device).to(device)
    optim_base, optim_tc = build_optimizer(base_model, tc_model, args)



    all_time = test_loss = test_accuracy = f1 = recall = 0

    scheduler_base = scheduler_tc = None

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    # load weight
    if "SupCon" == args.training_mode:
        load_pretrained_weight(base_model,args.training_mode,device,args, tc_model)
    elif "ft" in args.training_mode or "train_linear" in args.training_mode or "SupCon" in args.training_mode:
        load_pretrained_weight(base_model, args.training_mode, device, args)

    for epoch in range(args.epochs):
        if "gen_pseudo_labels" in args.training_mode:
            ft_perc = "1p"
            data_path = os.path.join(args.data_path)
            load_from = os.path.join(
                os.path.join(args.logs_save_dir, args.experiment_description, args.run_description, f"ft_{ft_perc}_seed_{args.seed}",
                             "best_models"))
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
            pretrained_dict = chkpoint["model_state_dict"]
            base_model.load_state_dict(pretrained_dict)
            # gen_pseudo_labels_V2(base_model, train_loader, device, data_path, args)
            gen_pseudo_labels(base_model, train_loader, device, data_path, args)
            sys.exit(0)

        logger.debug("Training started ....")

        if "ft_SupCon" in args.training_mode:
            train_accuracy, train_loss, epoch_time = train_epoch(base_model, tc_model, train_loader, optim_base,optim_tc, criterion, epoch, tb_writer, args, device)

            val_accuracy, vali_loss = val_epoch(base_model, tc_model, val_loader, criterion, epoch, tb_writer, args,
                                                device)

            test_accuracy, test_loss, f1, recall = test_epoch(base_model, test_loader, criterion, epoch, tb_writer,
                                                              args, device)

        elif "self_supervised" in args.training_mode or "SupCon" in args.training_mode:
            train_loss, epoch_time = train_epoch(base_model, tc_model, train_loader, optim_base, optim_tc, criterion, epoch, tb_writer, args, device)
            val_accuracy = 0
            train_accuracy = 0

            vali_loss = val_epoch(base_model, tc_model, val_loader, criterion, epoch, tb_writer, args, device)

            early_stopping(vali_loss, base_model, args.early_stopping_path)

        else:
            train_accuracy, train_loss, epoch_time = train_epoch(base_model, tc_model, train_loader, optim_base, optim_tc, criterion, epoch, tb_writer, args, device)

            val_accuracy, vali_loss = val_epoch(base_model, tc_model, val_loader, criterion, epoch, tb_writer, args, device)

            test_accuracy, test_loss, f1, recall = test_epoch(base_model, test_loader, criterion, epoch, tb_writer,
                                              args, device)

        if args.use_adj:
            adjust_learning_rate_sche(optim_tc, optim_base, epoch, args, scheduler_tc=scheduler_tc,
                                      scheduler_base=scheduler_base, tb_writer=tb_writer)

        eval_parameters = (train_loss, train_accuracy, vali_loss, val_accuracy, test_loss, test_accuracy, f1, recall)
        early_stopping(vali_loss, base_model, args.early_stopping_path)
        model_evaluate.evaluate(base_model, tc_model, eval_parameters)
        if args.use_earlystop:
              if early_stopping.early_stop:
                  model_evaluate.final_logger(logger)
                  break

        model_evaluate.logger(logger, eval_parameters, epoch)

    model_evaluate.final_logger(logger)



def show_feature(args):
    # import socket
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)
    # sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60 * 1000, 30 * 1000))

    # setting the seed and logger
    logger, experiment_log_dir = log_setting(args)
    # load data
    loader, data_size, dataset = load_data(args.data_type, args.data_path, args.seq_len, args.training_mode, args.batch_size,
                                  args)
    logger.debug("Data loaded ...")
    train_dataset, val_dataset, test_dataset = dataset

    base_model = model_selection(args)

    # base_model.to(device)
    if "ft" in args.training_mode or "train_linear" in args.training_mode or "SupCon" in args.training_mode:
        load_pretrained_weight(base_model, args.training_mode, device, args)
    feature_show_distance(base_model, test_dataset)


if __name__ == "__main__":
    runs(args)
    # show_feature(args)