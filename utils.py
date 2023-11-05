import os
import random
from math import pi, cos
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, confusion_matrix, f1_score, recall_score
import pandas as pd
from aug import masked
from cross_models.cross_former_model import Crossformer
from model.LSTM import MyLSTM
import time
import torch
from tqdm import tqdm
from torch import nn as nn
from loss import SupConLoss, NTXentLoss
import torch.nn.functional as F
import logging
import sys
from datetime import datetime
from model.baseModel import base_Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth') todo
        self.val_loss_min = val_loss


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    @property
    def mask(self):
        return self._mask


class Model_evaluation():
    def __init__(self, args, experiment_log_dir):
        self.min_val_loss = 200
        self.max_val_accuracy = 0
        self.max_recall = 0
        self.min_test_loss = 200
        self.max_f1 = 0
        self.train_loss = 0
        self.train_accuracy = 0
        self.max_test_accuracy = 0
        self.use_earlystop = args.use_earlystop
        self.training_mode = args.training_mode
        self.eval_ind = 200
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            self.eval_ind = 200
        else:
            self.eval_ind = 0
        self.experiment_log_dir = experiment_log_dir
        os.makedirs(os.path.join(self.experiment_log_dir, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_log_dir, "best_models"), exist_ok=True)

    def evaluate(self, base_model: nn.Module, tc_model: nn.Module, eval_parameters):
        train_loss, train_accuracy, vali_loss, val_accuracy, test_loss, test_accuracy, f1, recall = eval_parameters
        eval_ind = vali_loss
        if self.training_mode == "self_supervised" or self.training_mode =="SupCon":
            eval_ind = vali_loss
        # else:
        #     eval_ind = -val_accuracy
        self.save_weight(base_model, tc_model)
        if self.eval_ind > eval_ind:
            self.eval_ind = eval_ind
            self.train_loss = train_loss
            self.train_accuracy = train_accuracy
            self.min_val_loss = vali_loss
            self.max_val_accuracy = val_accuracy
            self.min_test_loss = test_loss
            self.max_test_accuracy = test_accuracy
            self.max_f1 = f1
            self.max_recall = recall
            self.save_weight(base_model, tc_model, save_type="best_models")

    def logger(self, logger, eval_parameters, epoch):
        train_loss, train_accuracy, vali_loss, val_accuracy, test_loss, test_accuracy, f1, recall = eval_parameters
        logger.debug(f'\nepoch: {epoch} '
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_accuracy:2.4f}\n'
                     f'Valid Loss     : {vali_loss:2.4f}\t | \tValid Accuracy     : {val_accuracy:2.4f}\n'
                     f'Test_Loss    : {test_loss:2.4f}\t | \tTest Accuracy     : {test_accuracy:2.4f}\n')

    def final_logger(self,logger):
        logger.debug(f'\nfinal '
                     f'Train Loss     : {self.train_loss:2.4f}\t | \tTrain Accuracy     : {self.train_accuracy:2.4f}\n'
                     f'Valid Loss     : {self.min_val_loss:2.4f}\t | \tValid Accuracy     : {self.max_val_accuracy:2.4f}\n'
                     f'Test_Loss    : {self.min_test_loss:2.4f}\t | \tTest Accuracy     : {self.max_test_accuracy:2.4f}\n'
                     f'F1      : {self.max_f1:2.4f}\t | \tRecall      : {self.max_recall:2.4f}')
        logger.debug("\n################## Training is Done! #########################")

    def save_weight(self, base_model: nn.Module, tc_model: nn.Module, save_type: str = "saved_models"):
        chkpoint = {'model_state_dict': base_model.state_dict(),
                    'temporal_contr_model_state_dict': tc_model.state_dict()}
        torch.save(chkpoint, os.path.join(self.experiment_log_dir, save_type, f'ckp_last.pt'))


def build_optimizer(base_model: nn.Module, TC_model: nn.Module, args):
    if args.lradj == "weight_decay":
        optimizer_base = torch.optim.Adam(base_model.parameters(), args.learning_rate, (args.beta1, args.beta2),
                                          weight_decay=args.weight_decay)

        optimizer_tc = torch.optim.Adam(TC_model.parameters(), args.learning_rate, (args.beta1, args.beta2),
                                          weight_decay=args.weight_decay)

    else:
        optimizer_base = torch.optim.Adam(base_model.parameters(), args.learning_rate, (args.beta1, args.beta2))

        optimizer_tc = torch.optim.Adam(TC_model.parameters(), args.learning_rate, (args.beta1, args.beta2))

    return optimizer_base, optimizer_tc


def model_selection(args):
    if args.model_type == "Crossformer":
        model = Crossformer(args.data_dim, args.seq_len, args.out_len, args.seg_len, win_size=args.merge_win_size, d_model=args.d_model,
                            n_heads=args.n_heads, e_layers=args.e_layers, dropout=args.dropout, num_classes=args.num_classes, logistic_dropout=args.logistic_dropout)
        return model
    elif args.model_type =="LSTM":
        model = MyLSTM(args.data_dim, args.hidden_size, args.num_classes, args.e_layers, args.use_bir)
        return model
    elif args.model_type == "CNN":
        model = base_Model(args)
        return model


def show_grad(model:nn.Module):
    for item, parms  in model.named_parameters():
        h = parms.grad


def self_Supervised_loss(features1, features2, tc_model: nn.Module, args, device, origin_data):
    nt_xent_criterion = NTXentLoss(device, args.batch_size, args.temperature,
                                   args.use_cosine_similarity)
    if args.loss_type == "c":
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        temp_cont_loss1, temp_cont_feat1 = tc_model(features1, features2)
        temp_cont_loss2, temp_cont_feat2 = tc_model(features2, features1)
        zjs = temp_cont_feat1
        zis = temp_cont_feat2
        loss = nt_xent_criterion(zjs, zis)
        return loss
    elif args.loss_type == "tc":
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        feature_data = F.normalize(origin_data, dim=1)
        # temp_cont_loss1, temp_cont_feat1 = tc_model(features1, features2)
        temp_cont_loss1, temp_cont_feat1 = tc_model(features1, feature_data)
        temp_cont_loss2, temp_cont_feat2 = tc_model(features2, feature_data)
        # temp_cont_loss2, temp_cont_feat2 = tc_model(features2, features1)
        loss = (temp_cont_loss1 + temp_cont_loss2) * args.lambda1 + nt_xent_criterion(temp_cont_feat1,
                                                                                 temp_cont_feat2) * args.lambda2
        return loss
    elif args.loss_type == "mask":
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        feature_data = F.normalize(origin_data, dim=1)
        # temp_cont_loss1, temp_cont_feat1 = tc_model(features1, features2)
        temp_cont_loss1, temp_cont_feat1 = tc_model(features1, feature_data)
        temp_cont_loss2, temp_cont_feat2 = tc_model(features2, feature_data)
        # temp_cont_loss2, temp_cont_feat2 = tc_model(features2, features1)
        loss = (temp_cont_loss1 + temp_cont_loss2)
        return loss


def train_epoch(base_model: nn.Module,tc_model: nn.Module, train_loader, optimizer_base, temp_cont_optimizer, criterion, epoch, tb_writer, args, device):
    train_bar = tqdm(train_loader)
    base_model.train()
    if args.training_mode =="self_supervised" or args.training_mode == "SupCon":
        tc_model.train()
    epoch_time = time.time()
    train_loss = []
    cos_similarity_loss = []
    train_mae = []
    total_acc =[]
    for step, info in enumerate(train_bar):
        if "comb" in args.data_aug_mode and "self_supervised" in args.training_mode:
            data, label, aug1, aug2, aug3, aug3_mask = info
            data, aug1, aug2, aug3, aug3_mask = data.to(device), aug1.to(device), aug2.to(device), aug3.to(device), aug3_mask.to(device)
        else:
            data, label, aug1, aug2 = info
            data, aug1, aug2 = data.to(device), aug1.to(device), aug2.to(device)
        if args.model_type == "CNN":
            if "comb" in args.data_aug_mode and "self_supervised" in args.training_mode :
                aug1, aug2,aug3 = aug1.permute(0, 2, 1), aug2.permute(0, 2, 1), aug3.permute(0,2,1)
            else:
                aug1, aug2 = aug1.permute(0, 2, 1), aug2.permute(0, 2, 1)
        else:
            data = data.permute(0, 2, 1)
        # data = data.permute(1,0,2)
        optimizer_base.zero_grad()
        if "self_supervised" in args.training_mode or "SupCon" in args.training_mode:
            temp_cont_optimizer.zero_grad()
            c_bs = data.shape[0]
            ## todo 三层对比+损失比例
            predictions4, features4 = base_model(data)
            predictions1, features1 = base_model(aug1)
            predictions2, features2 = base_model(aug2)
            # if args.use_mask:
            mask1 = masked(features1, args.mask_mode, args.mask_type)
            mask2 = masked(features2, args.mask_mode, args.mask_type)
            features1[~mask1] = 0
            masked_aug1 = features1
            masked_aug2 = features2
            features2[~mask2] = 0

        if "ft_SupCon" in args.training_mode:
            output = base_model(data)
            # label = torch.LongTensor(label)
            label = label.to(device)
            label = label.reshape(-1)
            predictions, features = output
            loss = criterion(predictions, label)
            train_loss.append(loss.detach().cpu())
            total_acc.append(label.eq(predictions.detach().argmax(dim=1)).float().mean())

        elif "self_supervised" in args.training_mode:
            loss = self_Supervised_loss(features1, features2, tc_model, args, device, features4)
            # cos_similarity_loss.append(nt_xent_criterion(temp_cont_feat1,temp_cont_feat2).detach().float().cpu())

        elif "SupCon" in args.training_mode:
            lambda1 = 0.01
            lambda2 = 0.1
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            feature_data = F.normalize(features4, dim=1)
            # temp_cont_loss1, temp_cont_feat1 = tc_model(features1, features2)
            temp_cont_loss1, temp_cont_feat1 = tc_model(features1, feature_data)
            temp_cont_loss2, temp_cont_feat2 = tc_model(features2, feature_data)
            _,temp_ori_feat = tc_model(feature_data,feature_data)
            Sup_contrastive_criterion = SupConLoss(device)
            # Sup_contrastive_criterion = SupConLossV2(device)

            # supCon_features = temp_ori_feat.unsqueeze(1)
            supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
            # loss = Sup_contrastive_criterion(supCon_features, label)
            #
            label = label.to(device)
            # loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + criterion(predictions4, label)*lambda2
            # loss = criterion(predictions4, label)

            #
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                             label) * lambda2
        else:
            output = base_model(data)
            label = torch.LongTensor(label)
            label = label.to(device)
            label = label.reshape(-1)
            predictions, features = output
            loss = criterion(predictions, label)
            total_acc.append(label.eq(predictions.detach().argmax(dim=1)).float().mean())

        train_loss.append(loss.item())
        loss.backward()

        show_grad(base_model)
        optimizer_base.step()
        if "self_supervised" in args.training_mode or "SupCon" in args.training_mode:
            temp_cont_optimizer.step()

        train_bar.desc = "[epoch {}] epoch loss {}  mean_loss {}".format(epoch + 1, round(loss.item(), 3),
                                                                         round(np.average(train_loss), 3))
    epoch_time = time.time() - epoch_time
    train_accuracy = torch.tensor(total_acc).mean()
    train_mae = torch.tensor(train_mae).mean()
    tb_writer.add_scalar("train_loss", torch.tensor(train_loss).mean(), epoch)
    if "ft_SupCon" in args.training_mode:
        tb_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        return train_accuracy, np.average(train_loss), epoch_time

    elif "self_supervised" in args.training_mode or "SupCon" in args.training_mode:
        tb_writer.add_scalar("train_cos_similarity_loss", torch.tensor(cos_similarity_loss).mean(), epoch)

        return np.average(train_loss), epoch_time
    else:
        tb_writer.add_scalar("train_accuracy", train_accuracy, epoch)
    return train_accuracy, np.average(train_loss), epoch_time



## macro-F1
def val_epoch(base_model: nn.Module, tc_model: nn.Module, val_loader, criterion, epoch, tb_writer, args, device):
    with torch.no_grad():
        base_model.eval()
        if args.training_mode == "self_supervised" or args.training_mode == "SupCon":
            tc_model.eval()
        val_loss = []
        total_acc = []
        all_label = []
        all_pred = []
        cos_similarity_loss = []
        val_bar = tqdm(val_loader)
        for step, info in enumerate(val_bar):

            if "ft_SupCon" in args.training_mode:
                data, label, _, _ = info
                if args.model_type != "CNN":
                    data = data.permute(0, 2, 1)

                data, label = data.to(device), label.to(device)
                label = torch.LongTensor(label)
                pred, feature = base_model(data)
                label = label.reshape(-1)
                total_acc.append(label.eq(pred.detach().argmax(dim=1)).float().mean())
                loss = criterion(pred, label)
                val_loss.append(loss.cpu())

            elif "self_supervised" in args.training_mode or "SupCon" in args.training_mode:
                if "comb" in args.data_aug_mode:
                    data, label, aug1, aug2, aug3, aug3_mask = info
                    data, aug1, aug2, aug3, aug3_mask = data.to(device), aug1.to(device), aug2.to(device), aug3.to(device), aug3_mask.to(device)
                else:
                    data, label, aug1, aug2 = info
                    data, aug1, aug2 = data.to(device), aug1.to(device), aug2.to(device)

                if args.model_type == "CNN":
                    if "comb" in args.data_aug_mode and "self_supervised" in args.training_mode:
                        aug1, aug2, aug3 = aug1.permute(0, 2, 1), aug2.permute(0, 2, 1),aug3.permute(0,2,1)

                    else:
                        aug1, aug2 = aug1.permute(0, 2, 1), aug2.permute(0, 2, 1)
                else:
                    data = data.permute(0, 2, 1)

                predictions1, features1 = base_model(aug1)
                predictions2, features2 = base_model(aug2)
                predictions4, features4 = base_model(data)

                if "self_supervised" in args.training_mode:
                    mask1 = masked(features1, args.mask_mode, args.mask_type)
                    mask2 = masked(features2, args.mask_mode, args.mask_type)
                    features1[~mask1] = 0
                    masked_aug1 = features1
                    features2[~mask2] = 0
                    masked_aug2 = features2


                loss = self_Supervised_loss(features1, features2, tc_model, args, device,features4)

                val_loss.append(loss.float().cpu())
                cos_similarity_loss.append(loss)

            else:
                data, label, _, _ = info
                if args.model_type != "CNN":
                    data = data.permute(0, 2, 1)

                data, label = data.to(device), label.to(device)
                label = torch.LongTensor(label)
                label=label.reshape(-1)
                pred, feature = base_model(data)

                total_acc.append(label.eq(pred.detach().argmax(dim=1)).float().mean())


                loss = criterion(pred, label)
                val_loss.append(loss.cpu())

        tb_writer.add_scalar("val_loss", torch.tensor(val_loss).mean(), epoch)

        if "ft_SupCon" in args.training_mode:
            val_accuracy = torch.tensor(total_acc).mean()
            val_accuracy = val_accuracy.detach().cpu().numpy()
            val_bar.desc = "[epoch {}] val_loss {}  accuracy {}".format(epoch + 1, np.average(val_loss), val_accuracy)
            tb_writer.add_scalar("val_accuracy", val_accuracy, epoch)
            return val_accuracy, np.average(val_loss)

        if "self_supervised" in args.training_mode or "SupCon" in args.training_mode:
            tb_writer.add_scalar("val_cos_similarity_loss", torch.tensor(cos_similarity_loss).mean(), epoch)
            # tb_writer.add_scalar("val_temp_features_1_loss", torch.tensor(temp_features_1_loss).mean(), epoch)
            # tb_writer.add_scalar("val_temp_features_2_loss", torch.tensor(temp_features_2_loss).mean(), epoch)
            val_bar.desc = "[epoch {}] val_loss {}".format(epoch + 1, np.average(val_loss))
            return np.average(val_loss)
        val_accuracy = torch.tensor(total_acc).mean()

        val_accuracy = val_accuracy.detach().cpu().numpy()
        # val_accuracy = train_mae.detach().cpu().numpy()
        val_bar.desc = "[epoch {}] val_loss {}  accuracy {}".format(epoch + 1, np.average(val_loss), val_accuracy)

        tb_writer.add_scalar("val_accuracy", val_accuracy, epoch)
        return val_accuracy, np.average(val_loss)


def test_epoch(model: nn.Module, test_loader, criterion, epoch, tb_writer, args, device):
    with torch.no_grad():
        model.eval()
        test_loss = []
        all_label = []
        all_pred = []
        total_acc = []
        train_mae=[]
        test_bar = tqdm(test_loader)
        for step, info in enumerate(test_bar):
            data, label, _, _ = info
            if args.model_type != "CNN":
                data = data.permute(0, 2, 1)
            data, label = data.to(device), label.to(device)
            pred, feature = model(data)
            label = label.reshape(-1)
            loss = criterion(pred, label)
            test_loss.append(loss.cpu())
            total_acc.append(label.eq(pred.detach().argmax(dim=1)).float().mean())

            pred = torch.max(pred, dim=1)[1]
            all_label.extend(list(label.cpu().numpy()))
            all_pred.extend(list(pred.cpu().numpy()))

        all_pred = np.array(all_pred)
        all_label = np.array(all_label)
        test_accuracy = torch.tensor(total_acc).mean()
        test_accuracy = test_accuracy.detach().cpu().numpy()
        tb_writer.add_scalar("test_accuracy", test_accuracy, epoch)
        tb_writer.add_scalar("test_loss", np.average(test_loss), epoch)
        f1=f1_score(all_label.reshape(-1), all_pred.reshape(-1), average="macro")
        recall = recall_score(all_label.reshape(-1), all_pred.reshape(-1), average="weighted")
    return test_accuracy, np.average(test_loss), f1, recall


def load_pretrained_weight(model: nn.Module, training_mode, device, args,tc_model:nn.Module = None):
    if "train_linear" in training_mode:
        if "SupCon" not in training_mode:
            load_from = os.path.join(
                os.path.join(args.logs_save_dir, args.experiment_description, args.run_description, f"self_supervised_seed_{args.seed}",
                             "best_models"))
        else:
            load_from = os.path.join(
                os.path.join(args.logs_save_dir, args.experiment_description, args.run_description, f"SupCon_seed_{args.seed}",
                             "best_models"))
    elif "ft" in training_mode:
        if "SupCon" not in training_mode:
            data_perc = 1
            load_from = os.path.join(
                os.path.join(args.logs_save_dir, args.experiment_description, args.run_description, f"self_supervised_seed_{args.seed}",
                             "best_models"))
        else:
            load_from = os.path.join(
                os.path.join(args.logs_save_dir, args.experiment_description, args.run_description, f"SupCon_seed_{args.seed}",
                             "best_models"))

    elif "SupCon" in training_mode:
        data_perc = 1

        load_from = os.path.join(
            os.path.join(args.logs_save_dir, args.experiment_description, args.run_description,
                     f"self_supervised_seed_{args.seed}",
                     "best_models"))

    elif "gen_pseudo_labels" in training_mode:
        ft_perc = "1p"
        load_from = os.path.join(
            os.path.join(args.logs_save_dir, args.experiment_description, args.run_description, f"ft_{ft_perc}_seed_{args.seed}",
                         "best_models"))

    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]


    # pretrained_dict = chkpoint
    if "SupCon" == training_mode:
        model.load_state_dict(pretrained_dict)
        # tc_weight = chkpoint["temporal_contr_model_state_dict"]
        # tc_model.load_state_dict(tc_weight)

    elif "ft" or "train_linear" in training_mode:
        model_dict = model.state_dict()
        del_list = ['logits']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if "train_linear" in training_mode:
        set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.


def gen_pseudo_labels(model, dataloader, device, experiment_log_dir, args):
    model.eval()
    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []
    total_acc=[]

    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            data = data.float().to(device)
            # data = data.permute(0, 2, 1)
            # labels = labels.view((-1)).long().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            predictions, features = model(data)

            normalized_preds = softmax(predictions)
            total_acc.append(labels.eq(normalized_preds.detach().argmax(dim=1)).float().mean())
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            # all_labels = np.append(all_labels, labels.cpu().numpy())
            # data = data.permute(0, 2, 1)
            all_data.append(data)

    all_data = torch.cat(all_data, dim=0)
    val_accuracy = torch.tensor(total_acc).mean()

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_data.pt"
    torch.save(data_save, os.path.join(os.path.join(experiment_log_dir, args.data_type), file_name))
    print("train_accuracy: "+str(val_accuracy))
    print("Pseudo labels generated ...")


def gen_pseudo_labels_V2(model, dataloader, device, experiment_log_dir, args):
    model.eval()
    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []
    total_acc = []

    with torch.no_grad():
        for data, labels, aug1, aug2 in dataloader:
            data = data.float().to(device)
            # data = data.permute(0, 2, 1)
            aug1 = aug1.permute(0,2,1).float().to(device)
            aug2 = aug2.permute(0,2,1).float().to(device)
            # labels = labels.view((-1)).long().to(device)
            labels = labels.view((-1)).long().to(device)
            # forward pass
            predictions, features = model(data)
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            normalized_preds = softmax(predictions)
            normalized_preds1 = softmax(predictions1)
            normalized_preds2 = softmax(predictions2)
            total_acc.append(labels.eq(normalized_preds.detach().argmax(dim=1)).float().mean())
            pseudo_labels_ori = normalized_preds.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()
            pseudo_labels1 = normalized_preds1.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()
            pseudo_labels2 = normalized_preds2.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()

            idx = [i for i in range(len(pseudo_labels1)) if pseudo_labels2[i] == pseudo_labels1[i]]

            pseudo_labels = pseudo_labels_ori[idx]
            ps_data = data.detach().cpu().numpy()[idx]

            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels)

            # all_labels = np.append(all_labels, labels.cpu().numpy())
            # data = data.permute(0, 2, 1)
            all_data.append(torch.FloatTensor(ps_data))

    all_data = torch.cat(all_data, dim=0)
    val_accuracy = torch.tensor(total_acc).mean()

    data_save = dict()
    data_save["samples"] = torch.FloatTensor(all_data)
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_data.pt"
    torch.save(data_save, os.path.join(os.path.join(experiment_log_dir, args.data_type), file_name))
    print("train_accuracy: " + str(val_accuracy))
    print("Pseudo labels generated ...")

def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def seed_setting(SEED):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)


def adjust_learning_rate_sche(optim_tc, optim_base, epoch, args, scheduler_tc=None, scheduler_base=None, tb_writer=None):
    if args.lradj == "warm_up_cosine":
        if epoch < args.warm_up_step:
            lr = epoch * args.max_learning_rate / args.warm_up_step
            # for param_group in optim_tc.param_groups:
            #     param_group['lr'] = lr
            for param_group in optim_base.param_groups:
                param_group['lr'] = lr
        else:
            lr = args.lr_min + (args.max_learning_rate - args.min_learning_rate) * (
                        1 + cos(
                    pi * (epoch - args.warm_up_step) / (args.epochs - args.warm_up_step))) / 2
            # for param_group in optim_tc.param_groups:
            #     param_group['lr'] = lr
            for param_group in optim_base.param_groups:
                param_group['lr'] = lr

    elif args.lradj == "Cosine_annealing":
        scheduler_tc.step()
        lr = scheduler_tc.get_lr()
        scheduler_base.step()

    elif args.lradj == "normal":
        lr = args.learning_rate * (0.1 ** (epoch // 50))
        for param_group in scheduler_base.param_groups:
            param_group['lr'] = lr
        for param_group in scheduler_tc.param_groups:
            param_group['lr'] = lr

    elif args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optim_tc.param_groups:
                param_group['lr'] = lr
            for param_group in optim_tc.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optim_tc.param_groups:
                param_group['lr'] = lr
            for param_group in optim_tc.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))


    elif args.lradj =="linear_decay":
        lr = args.lr_min + (args.max_learning_rate - args.min_learning_rate) * (
                1 + cos(
            pi * (epoch - 0) / (args.epochs -0))) / 2
        # for param_group in optim_tc.param_groups:
        #     param_group['lr'] = lr
        for param_group in optim_base.param_groups:
            param_group['lr'] = lr

    # tb_writer.add_scalar("optim_tc_lr", lr, epoch)
    tb_writer.add_scalar("optim_base_lr", lr, epoch)


def log_setting(args):
    seed_setting(args.seed)

    experiment_description = args.experiment_description
    training_mode = args.training_mode
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_seed_{args.seed}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.data_type}')
    if args.use_earlystop :
        logger.debug(f'Mode:    {args.training_mode}   early_stopping:   {args.patience}    lr:   {args.learning_rate}')
    else:
        logger.debug(f'Mode:    {args.training_mode}   epochs:   {args.epochs}    lr:   {args.learning_rate}  logitis drop {args.logistic_dropout}  temperature: {args.temperature}')
    logger.debug("=" * 45)

    return logger, experiment_log_dir


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def feature_show_distance(model: nn.Module, data, mode: str = "pca"):
    with torch.no_grad():
        model.eval()
        sample = data["samples"]
        label = data["labels"]
        # sample.to(device)
        # sample = sample.permute(0,2,1)
        # sample=sample.reshape(sample.shape[0],1,-1)
        # sample=sample.permute(0,2,1)
        sample=sample.numpy()
        label=label.numpy()
        sample = torch.FloatTensor(sample)
        pred, feature = model(sample)
        feature = feature.reshape(sample.shape[0],-1)

    if mode == "pca":
        pca = PCA(n_components=2)
        pca.fit(feature)
        X_dr = pca.transform(feature)
        # 鸢尾花三种类型可视化
        colors = ['red', 'black', 'orange', 'green', 'blue', 'blueviolet', 'gold']

        plt.figure()
        for i in range(X_dr.shape[0]):
            plt.scatter(x=X_dr[i][0],y=X_dr[i][1]
                        , alpha=.7
                        , c=colors[int(label[i])])
        # plt.legend()
        plt.title('feature of CNN')
        plt.show()

    elif mode == "tsne":
        tsne = TSNE(n_components=2, perplexity=50,n_iter=5000,n_jobs=10)
        X_dr = tsne.fit(feature).embedding_
        colors = ['red', 'black', 'orange', 'green', 'blue', 'blueviolet', 'gold']

        plt.figure()
        for i in range(X_dr.shape[0]):
            plt.scatter(x=X_dr[i][0], y=X_dr[i][1]
                        , alpha=.7
                        , c=colors[int(label[i])])
        plt.legend()
        plt.title('feature of CNN')
        plt.show()


def val_data_split_label(test_dataset):
    test_data, test_label = test_dataset
    dr1=dr2=dr3=dr4=dr5=dr6=[]
    for i in range(len(test_label)):
        if test_label[i] ==0:
            dr1.append(test_data[i])
        elif test_label[i]==1:
            dr2.append(test_data[i])

        elif test_label[i]==2:
            dr3.append(test_data[i])

        elif test_label[i]==3:
            dr4.append(test_data[i])

        elif test_label[i]==4:
            dr5.append(test_data[i])

        elif test_label[i]==5:
            dr6.append(test_data[i])
    dr1=torch.Tensor(dr1)
    dr2=torch.Tensor(dr2)
    dr3=torch.Tensor(dr3)
    dr4=torch.Tensor(dr4)
    dr5=torch.Tensor(dr5)
    dr6=torch.Tensor(dr6)
    return (dr1,dr2,dr3,dr4,dr5,dr6)


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]