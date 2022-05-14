import matplotlib.pyplot as plt
import numpy as np
from ray import tune
import torch
from torch import nn
from torch import optim

from models import AttentionRNN


def train_net(model, criterion, optimizer, num_epochs, train_loader, test_x, test_y, device):

    train_losses = []
    test_losses = []

    for _ in range(num_epochs):
        runnning_loss = 0.0
        idx = None
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # forward
            score_y = model(batch_x)
            loss = criterion(score_y, batch_y)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update params
            optimizer.step()

            # add train loss
            runnning_loss += loss.item()
        train_losses.append(runnning_loss / idx)

        # add test loss
        model.eval()
        score_y = model(test_x)
        test_loss = criterion(score_y, test_y)
        test_losses.append(test_loss.item())

    # plot loss curve
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")

    # save state_dict
    torch.save(model.state_dict(), "./mlp.pth")

    return model

"""
def train_timeseries_net(*, model, criterion, optimizer, train_loader,
                         test_x, test_y, device, patience, num_epochs=50):
    train_losses = []
    test_losses = []
    early_stopping = EarlyStopping(patience=patience)

    for _ in range(num_epochs):
        model.train()
        runnning_loss = 0.0
        idx = None
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # forward
            score_y = model(batch_x, device)
            score_y = torch.sigmoid(score_y.reshape(-1))
            batch_y = batch_y.reshape(-1)
            loss = criterion(score_y, batch_y)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # add training loss
            runnning_loss += loss.item()
        train_losses.append(runnning_loss / idx)

        # add test loss
        model.eval()
        pred_y = model(test_x, device)
        pred_y = torch.sigmoid(pred_y.reshape(-1))
        test_y = test_y.reshape(-1)
        test_loss = criterion(pred_y, test_y)
        test_losses.append(test_loss.item())

        # early stopping
        early_stopping(test_loss)
        if early_stopping.early_stop:
            break

    # plot loss curve
    plt.plot(train_losses, label="train", alpha=0.5, c="r")
    plt.plot(test_losses, label="test", alpha=0.5, c="b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.savefig("loss.png")

    # save state_dict
    torch.save(model.state_dict(), "./rnn.pth")
    return model
"""

def train_seq2seq_net(model, criterion, optimizer,
                      num_epochs, train_loader, device):
    train_losses = []

    for _ in range(num_epochs):
        runnning_loss = 0.0
        idx = None
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            decoder_y = batch_y[:, :-1, :]
            decoder_t = batch_y[:, 1:, :].to(torch.long)

            # forward
            score_y = model(batch_x, decoder_y, device)
            loss = criterion(score_y.reshape(-1, 2), decoder_t.reshape(-1))

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # add training loss
            runnning_loss += loss.item()
        train_losses.append(runnning_loss / idx)

    # plot loss curve
    plt.plot(train_losses, label="train", alpha=0.5, c="r")
    plt.ylim(0.2, 1.0)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")

    # save state_dict
    torch.save(model.state_dict(), "./seq2seq.pth")
    return model


class EarlyStopping:

    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_timeseries_net(config, options):

    # 1. assign data and some variables from options
    train_loader, val_x, val_y = options["dataset"].values()
    input_size, num_layers, num_classes = options["params"].values()
    num_epochs = options["num_epochs"]
    device = options["device"]

    # 2. instantiate model, criterion, optimizer
    model = AttentionRNN(input_size=input_size, hidden_size=config["hidden_size"],
                         num_layers=num_layers, num_classes=num_classes,
                         fc_sizes=[config["fc_size_0"], config["fc_size_1"], config["fc_size_2"]]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"], eps=config["eps"])
    early_stopping = EarlyStopping(patience=10)

    # 3. training loop
    for epoch in range(num_epochs):
        model.train()

        for _, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 3.1 forward
            score_y = model(batch_x, device)
            score_y = torch.sigmoid(score_y.reshape(-1))
            batch_y = batch_y.reshape(-1)
            loss = criterion(score_y, batch_y)

            # 3.2 backward
            optimizer.zero_grad()
            loss.backward()

            # 3.3 update params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # 3.4 add test loss
        model.eval()
        pred_y = model(val_x, device)
        pred_y = torch.sigmoid(pred_y.reshape(-1))
        val_y = val_y.reshape(-1)
        test_loss = criterion(pred_y, val_y)
        tune.report(loss=test_loss.item())
        
        # 3.5 save model's state_dict
        if epoch&5 == 0:
            torch.save(model.state_dict(), "./rnn.pth")

        # 3.6 early stopping
        early_stopping(test_loss.item())
        if early_stopping.early_stop:
            break