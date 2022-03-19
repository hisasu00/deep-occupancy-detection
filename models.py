from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, p):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 15)
        self.fc3 = nn.Linear(15, num_classes)
        self.bn1 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.bn1(x)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
