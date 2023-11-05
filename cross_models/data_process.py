import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
from typing import List
import numpy as np
from aug import DataTransform
import pandas as pd


class MyDataSet(Dataset):
    def __init__(self, datasource, args):
        self.data = datasource["samples"]
        if args.data_type == "pFD":
            self.data=self.data.reshape(self.data.shape[0], 1, -1)
        self.use_self_supervised = False
        self.use_Sup_con = False
        self.use_gen = False
        self.data_aug_mode= args.data_aug_mode
        if "gen" in args.training_mode:
            if "comb" in args.data_aug_mode:
                self.aug1, self.aug2,_ = DataTransform(self.data, args)
            else:
                self.aug1, self.aug2 = DataTransform(self.data, args)
            self.use_gen = True
        if "SupCon" in args.training_mode:
            self.label = datasource["labels"]
            self.use_Sup_con = True
        if "ft" in args.training_mode or "train_linear" in args.training_mode or "supervised_1p" in args.training_mode or "gen_pseudo_labels" in args.training_mode:
            self.label = datasource["labels"]
            self.use_self_supervised = False
        self.flag = False
        if "self_supervised" in args.training_mode or "SupCon" in args.training_mode:  # no need to apply Augmentations in other modes
            # self.data_t = np.transpose(self.data, (0, 2, 1))
            if "comb" in args.data_aug_mode:
                self.aug1, self.aug2, (self.aug3, self.aug3_mask) = DataTransform(self.data,args)
            else:
                self.aug1, self.aug2 = DataTransform(self.data, args)
                self.flag = True
            self.use_self_supervised = True

    def __getitem__(self, idx):
        single_data = self.data[idx]
        # single_data
        if isinstance(single_data, np.ndarray):
            data_t = single_data
        else:
            data_t = np.array(single_data.cpu(), dtype=np.float32)
        data_t = torch.FloatTensor(data_t)

        if self.use_Sup_con:
            aug1_t = np.array(self.aug1[idx], dtype=np.float32)
            aug2_t = np.array(self.aug2[idx], dtype=np.float32)
            single_label = self.label[idx]
            if "comb" in self.data_aug_mode:
                aug3_t_mask = np.array(self.aug3_mask,dtype=np.int)
                aug3_t = np.array(self.aug3[idx], dtype=np.float32)
                return data_t, single_label, torch.FloatTensor(aug1_t), torch.FloatTensor(aug2_t), torch.FloatTensor(aug3_t), torch.IntTensor(aug3_t_mask)
            return data_t, single_label, torch.FloatTensor(aug1_t), torch.FloatTensor(aug2_t)

        if self.use_self_supervised:
            aug1_t = np.array(self.aug1[idx], dtype=np.float32)
            aug2_t = np.array(self.aug2[idx], dtype=np.float32)
            if "comb" in self.data_aug_mode:
                aug3_t_mask = np.array(self.aug3_mask, dtype=np.int)
                aug3_t = np.array(self.aug3[idx], dtype=np.float32)
                return data_t, None, torch.FloatTensor(aug1_t), torch.FloatTensor(aug2_t), torch.FloatTensor(
                    aug3_t), torch.IntTensor(aug3_t_mask)
            return data_t, None, torch.FloatTensor(aug1_t), torch.FloatTensor(aug2_t)
        single_label = self.label[idx]

        if self.use_gen:
            aug1_t = np.array(self.aug1[idx], dtype=np.float32)
            aug2_t = np.array(self.aug2[idx], dtype=np.float32)
            return data_t, single_label, torch.FloatTensor(aug1_t), torch.FloatTensor(aug2_t)

        return data_t, single_label, data_t, data_t

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        data, labels, aug1, aug2 = tuple(zip(*batch))
        aug1 = torch.stack(aug1, dim=0)
        aug1 = aug1.permute(0, 2, 1)
        aug2 = torch.stack(aug2, dim=0)
        aug2 = aug2.permute(0, 2, 1)

        data = torch.stack(data, dim=0)
        # labels = torch.LongTensor(labels)
        # labels =torch.stack(labels, dim=0)

        labels = torch.LongTensor(torch.stack(labels, dim=0).numpy())
        return data, labels, aug1, aug2

    @staticmethod
    def collate_fn_self_supervised(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        data, _, aug1, aug2 = tuple(zip(*batch))
        aug1 = torch.stack(aug1, dim=0)
        aug1 = aug1.permute(0, 2, 1)
        aug2 = torch.stack(aug2, dim=0)
        aug2 = aug2.permute(0, 2, 1)
        data = torch.stack(data, dim=0)
        # labels = torch.as_tensor(labels)
        return data, None, aug1, aug2

    @staticmethod
    def collate_fn_self_supervisedv3(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        data, _, aug1, aug2, aug3,aug3_mask = tuple(zip(*batch))
        aug1 = torch.stack(aug1, dim=0)
        aug1 = aug1.permute(0, 2, 1)
        aug2 = torch.stack(aug2, dim=0)
        aug2 = aug2.permute(0, 2, 1)
        aug3 = torch.stack(aug3, dim=0)
        aug3 = aug3.permute(0, 2, 1)
        data = torch.stack(data, dim=0)
        aug3_mask = torch.stack(aug3_mask, dim=0)
        # labels = torch.as_tensor(labels)
        return data, None, aug1, aug2, aug3,aug3_mask


def label_process(labels, task_mode: str, threshold: List[float]):
    if task_mode == "classification":
        ps_label = []
        for label in labels:
            flag = 0
            for idx in range(len(threshold)):
                threshold_t = threshold[idx]
                if label <= threshold_t:
                    ps_label.append(idx)
                    flag = 1
                    break
            if flag == 0:
                ps_label.append((idx + 1))
        return np.array(ps_label)
    else:
        return labels


class MyDataSetPred(Dataset):
    def __init__(self, data_x,data_y, in_len ,out_len,args):
        self.data_x = data_x
        self.data_y = data_y
        self.in_len = in_len
        self.out_len = out_len
        self.self_supervised = False
        self.samples = []
        self.labels = []
        self.process_data()

        self.samples = self.samples.transpose(0,2,1)

        if "self_supervised" in args.training_mode:
            self.aug1, self.aug2 = DataTransform(self.samples, args)
            self.self_supervised = True
        self.in_len = in_len
        self.out_len = out_len

    def process_data(self):
        for index in range(len(self.data_x)-self.in_len-self.out_len+1):
            s_begin = index
            s_end = s_begin + self.in_len
            r_begin = s_end
            r_end = r_begin + self.out_len
            self.samples.append(self.data_x[s_begin:s_end])
            self.labels.append(self.data_y[r_begin:r_end])
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        if self.self_supervised:
            seq_x = self.aug1[index]
            ori_x = self.samples[index]
            seq_x2 = self.aug2[index]
            seq_y = self.labels[index]
            return ori_x,seq_y, seq_x, seq_x2,
        else:
            seq_x = self.samples[index]
            seq_y = self.labels[index]
        return seq_x, seq_y,seq_x,seq_x

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        data, labels, aug1, aug2 = tuple(zip(*batch))

        # aug1 = torch.stack(aug1,dim=0)
        aug1 = torch.FloatTensor(aug1)
        aug1 = aug1.permute(0, 2, 1)
        aug2 = torch.FloatTensor(aug2)
        aug2 = aug2.permute(0, 2, 1)

        data = torch.FloatTensor(data)
        # labels = torch.LongTensor(labels)
        # labels =torch.stack(labels, dim=0)

        labels = torch.FloatTensor(labels)
        return data, labels, aug1, aug2

def load_forecast_data(args,univar = False):

    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(args.data_type,univar)


    border1s = [0, train_slice - args.in_len, train_slice + valid_slice - args.in_len]
    border2s = [train_slice, train_slice + valid_slice, train_slice + valid_slice + test_slice]
    data = data.reshape(data.shape[1],-1)

    train_samples = data[border1s[0]:border2s[0]]
    train_labels = data[border1s[0]:border2s[0]]
    val_samples = data[border1s[1]:border2s[1]]
    val_labels = data[border1s[1]:border2s[1]]
    test_samples = data[border1s[2]:border2s[2]]
    test_labels= data[border1s[2]:border2s[2]]

    train_dataSet = MyDataSetPred(train_samples, train_labels, args.in_len, args.out_len,args)
    val_dataSet = MyDataSetPred(val_samples, val_labels, args.in_len, args.out_len,args)
    test_dataSet = MyDataSetPred(test_samples, test_labels, args.in_len, args.out_len,args)

    collat_fn_func = MyDataSetPred.collate_fn

    batch_size = args.batch_size

    if train_dataSet.__len__() < args.batch_size:
        batch_size = 16

    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,
                              num_workers=0,
                              collate_fn=collat_fn_func)
    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,
                            num_workers=0,
                            collate_fn=collat_fn_func)
    test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False,
                             num_workers=0,
                             collate_fn=collat_fn_func)
    return (train_loader, val_loader, test_loader), (len(train_dataSet), len(val_dataSet), len(test_dataSet)),(train_dataSet,val_dataSet,test_dataSet)



def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'data/forecast/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]

    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]

    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
        train_num = 12 * 30 * 24
        valid_num = 16 * 30 * 24  - train_num
        test_num = 20 * 30 * 24  - valid_num
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12 * 30 * 24 * 4)
        train_num = 12*30*24*4
        valid_num = 16*30*24*4 - train_num
        test_num = 20*30*24*4 - valid_num
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)

    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return data, train_num, valid_num, test_num, scaler, pred_lens, n_covariate_cols


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def load_data(data_type: str, default_data_path: str, seq_len: int, training_mode: str, batch_size: int, args):
    dir_path = None
    if data_type == "chemical_faber":
        dir_path = load_chemical_faber(data_type, default_data_path, seq_len, training_mode)
    elif data_type == "AReM":
        dir_path = os.path.join(default_data_path, "AReM")
    elif data_type == "Air":
        dir_path = os.path.join(default_data_path, "air")
    elif data_type == "HAR":
        dir_path = os.path.join(default_data_path, "HAR")
    elif data_type == "Epilepsy":
        dir_path = os.path.join(default_data_path, "Epilepsy")
    elif data_type == "sleepEDF":
        dir_path = os.path.join(default_data_path, "sleepEDF")

    elif data_type == "shiyan":
        dir_path = r"E:\MyCode\ChemicalFaber_new"
    elif data_type == "pFD":
        dir_path = os.path.join(default_data_path,"pFD")
    else:
        dir_path = os.path.join(default_data_path, data_type)

    if "all" in training_mode:
        train_dataset = torch.load(os.path.join(dir_path, "pseudo_train_data.pt"))
    elif "_1p" in training_mode:
        train_dataset = torch.load(os.path.join(dir_path, "train_1perc.pt"))
        # train_dataset = torch.load(os.path.join(dir_path, "pseudo_train_data.pt"))
    elif "_5p" in training_mode:
        train_dataset = torch.load(os.path.join(dir_path, "train_5perc.pt"))
    elif "_10p" in training_mode:
        train_dataset = torch.load(os.path.join(dir_path, "train_10perc.pt"))
    elif "_50p" in training_mode:
        train_dataset = torch.load(os.path.join(dir_path, "train_50perc.pt"))
    elif "_75p" in training_mode:
        train_dataset = torch.load(os.path.join(dir_path, "train_75perc.pt"))
    elif training_mode == "SupCon":
        train_dataset = torch.load(os.path.join(dir_path, "pseudo_train_data.pt"))
    else:
        train_dataset = torch.load(os.path.join(dir_path, "train.pt"))


    valid_dataset = torch.load(os.path.join(dir_path, "val.pt"))
    test_dataset = torch.load(os.path.join(dir_path, "test.pt"))

    train_dataSet = MyDataSet(train_dataset, args)
    val_dataSet = MyDataSet(valid_dataset, args)
    test_dataSet = MyDataSet(test_dataset, args)

    collat_fn_func = MyDataSet.collate_fn

    if "self_supervised" in args.training_mode:
        collat_fn_func = MyDataSet.collate_fn_self_supervised
        if "comb" in args.data_aug_mode:
            collat_fn_func = MyDataSet.collate_fn_self_supervisedv3

    if train_dataSet.__len__() < batch_size:
        batch_size = 16

    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True, num_workers=0,
                                  collate_fn=collat_fn_func)
    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True, num_workers=0,
                                collate_fn=collat_fn_func)
    test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=False, num_workers=0,
                                 collate_fn=collat_fn_func)

    return (train_loader, val_loader, test_loader), (len(train_dataSet), len(val_dataSet), len(test_dataSet)),(train_dataset,valid_dataset,test_dataset)


def load_chemical_faber(data_type: str, default_data_path: str, seq_len: int, training_mode: str):

    if "nt" in data_type:
        dir_path = os.path.join(default_data_path, "new_threshold")
    else:
        dir_path = os.path.join(default_data_path, "old_threshold")
    if "self_supervised" in training_mode:
        dir_path = os.path.join(dir_path, "unlabeled")
    elif "SupCon" in training_mode:
        dir_path = os.path.join(dir_path, "")
    else:
        dir_path = os.path.join(dir_path, "labeled")

    dir_path = os.path.join(dir_path, "win_size_" + str(seq_len))
    return dir_path



