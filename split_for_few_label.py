import numpy as np
import torch
import os
import random
from cross_models.data_process import label_process


def split_few_label_data(data_path: str, label_percent: int,  output_dir: str, cur_seed: int = 0, win_size: int = 15):
    """
    split the few label data
    :param data_path:
    :param label_percent:
    :return:
    """
    train_dataset = torch.load(os.path.join(data_path))
    X_train = train_dataset["samples"]
    y_train = train_dataset["labels"]
    all_size = X_train.shape[0]
    label_num = int(all_size*label_percent/100)
    idx = list(range(all_size))
    random.seed(cur_seed)
    random.shuffle(idx)
    label_idx = idx[:label_num]
    labeled_idx = [i for i in idx if i in label_idx]
    label_data = X_train[labeled_idx]
    label = y_train[labeled_idx]
    data_save = dict()
    data_save["samples"] = label_data
    data_save["labels"] = label
    torch.save(data_save, os.path.join(output_dir, f"train_{label_percent}perc.pt"))


def data_process(data_type: str, outdir: str, win_size: int):
    if data_type == "AReM":
        default_dir_path = "E:\MyCode\ChemicalFaber_new\data\AReM\win_size_50"
        file_path = os.path.join(default_dir_path, "AReM_"+str(win_size)+".pt")
        return torch.load(file_path)
    elif data_type == "ETT":
        default_dir_path = ""
        file_path = ""
    elif data_type == "chemical_faber":
        default_dir_path = r"E:\MyCode\ChemicalFaber_new\data\chemical_faber_threshold\avg\filter\win_size_50"
        file_path = os.path.join(default_dir_path, "avg_jn5_"+str(win_size)+".pt")
        return torch.load(file_path)
    else:
        pass


def split_data_process(datasource, data_type:str, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float= 0.2):
    sample = datasource["Sample"]
    label = datasource["labels"]
    # if data_type == "chemical_faber":
    #     label = label_process(label, "classification", [1.1, 1.3])
    data_size = len(sample)
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    test_size = data_size - train_size - val_size
    random.seed(0)
    data_idx = list(range(data_size))
    random.shuffle(data_idx)

    train_idx = data_idx[:train_size]
    val_idx = data_idx[train_size:train_size + val_size]
    test_idx = data_idx[train_size + val_size:]

    train_data, train_label = sample[train_idx], label[train_idx]
    val_data, val_label = sample[val_idx], label[val_idx]
    test_data, test_label = sample[test_idx], label[test_idx]

    train_dict = dict()
    train_dict["Sample"] = train_data
    train_dict["labels"] = train_label
    val_dict = dict()
    val_dict["Sample"] = val_data
    val_dict["labels"] = val_label
    test_dict = dict()
    test_dict["Sample"] = test_data
    test_dict["labels"] = test_label

    torch.save(train_dict, r"E:\MyCode\ChemicalFaber_new\train.pt")
    torch.save(val_dict, r"E:\MyCode\ChemicalFaber_new\val.pt")
    torch.save(test_dict, r"E:\MyCode\ChemicalFaber_new\test.pt")


def self_supervised_data_process(dir_path: str, win_size: int = 15, data_num: int = 5, train_ratio: float = 0.6, val_ratio: float = 0.2):
    data_list = []
    for idx in range(data_num):
        file_path = os.path.join(dir_path, "unlabeled_"+str(win_size)+"_29_"+str(idx+1)+".pt")
        data_t = torch.load(file_path)
        data_list.append(data_t["Sample"])
    all_data = np.concatenate(data_list, axis=0)
    data_size = len(all_data)

    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    test_size = data_size - train_size - val_size
    random.seed(0)
    data_idx = list(range(data_size))
    random.shuffle(data_idx)

    train_idx = data_idx[:train_size]
    val_idx = data_idx[train_size:train_size + val_size]
    test_idx = data_idx[train_size + val_size:]

    train_data = all_data[train_idx]
    val_data = all_data[val_idx]
    test_data = all_data[test_idx]

    train_dict = dict()
    train_dict["Sample"] = train_data
    # train_dict["labels"] = train_label
    val_dict = dict()
    val_dict["Sample"] = val_data
    # val_dict["labels"] = val_label
    test_dict = dict()
    test_dict["Sample"] = test_data
    # test_dict["labels"] = test_label

    torch.save(train_dict, r"/data/chemical_faber/new_threshold/unlabeled/avg/win_size_50\train.pt")
    torch.save(val_dict, r"/data/chemical_faber/new_threshold/unlabeled/avg/win_size_50\val.pt")
    torch.save(test_dict, r"/data/chemical_faber/new_threshold/unlabeled/avg/win_size_50\test.pt")


if __name__ == "__main__":
    # data_path = r"E:\MyCode\ChemicalFaber_new\data\chemical_fiber_processed\his\jn_25\win_size_50\train.pt"
    output_dir = r"E:\MyCode\ChemicalFaber_new\data\occupany"
    # data_type = "chemical_faber"
    dir_path = r"E:\MyCode\ChemicalFaber_new\data\occupany\train.pt"
    # data_source = data_process(data_type="chemical_faber", outdir=output_dir,win_size=50)
    # self_supervised_data_process(dir_path, win_size=15)
    # datasource = torch.load(r"E:\MyCode\ChemicalFaber_new\avg_process_20.pt")
    # split_data_process(datasource, None)
    few_lbl_percentages = [1, 5, 10, 50, 75]
    for few_lal in few_lbl_percentages:
        split_few_label_data(data_path=dir_path, label_percent=few_lal, output_dir=output_dir)




