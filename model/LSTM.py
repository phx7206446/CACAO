import torch.nn as nn
import torch
from torch import Tensor


class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, layer_num: int = 1, use_bir: bool = False):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.output_size=output_size
        self.feature_layer = nn.Linear(hidden_size, hidden_size//4)
        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(hidden_size//4,output_size),
                                        # nn.ReLU(),
                                        # nn.Dropout(),
                                        # nn.Linear(hidden_size//8,output_size)
                                        )
        # self.classifier = nn.Linear(hidden_size,output_size)

    def forward(self, inputs: Tensor):
        inputs = inputs.permute(1, 0, 2)
        out, (ct, ht) = self.lstm(inputs)
        feature = self.feature_layer(ht)
        # print("feature ______________")
        # print(feature)
        res = self.classifier(self.dropout(torch.relu(feature)))
        # print("res ______________________")
        # print(res)
        return res.reshape(-1, self.output_size), feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
