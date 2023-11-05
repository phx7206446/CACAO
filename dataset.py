from torch.utils.data import DataLoader, Dataset
from utils import DataTransform


class MyDataSet(Dataset):
    def __init__(self, dataSource, train_mode, config):
        super(MyDataSet, self).__init__()
        self.data = dataSource["Sample"]
        self.label = dataSource["labels"]
        self.train_mode = train_mode
        self.config = config
        if self.train_mode == "self_supervised" or self.train_mode:
            self.aug1, self.aug2 = DataTransform(self.x_data, self.config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[idx], self.y_data[idx], self.aug1[idx], self.aug2[idx]
        else:
            return self.x_data[idx], self.y_data[idx], self.x_data[idx], self.x_data[idx]



