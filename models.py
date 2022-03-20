import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout_ratio):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 15)
        self.fc3 = nn.Linear(15, num_classes)
        self.bn1 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.bn1(x)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_ratio):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 5)
        self.fc3 = nn.Linear(5, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)

        out, _ = self.rnn(x, (h0,c0))
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = F.relu(self.fc3(out))
        return out
