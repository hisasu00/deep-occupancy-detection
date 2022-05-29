import matplotlib.pyplot as plt
import numpy as np
import torch


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


def train_seq2seq_net(model, criterion, optimizer, num_epochs,
                      train_loader, test_x, test_y, device):
    train_losses = []
    test_losses = []

    for _ in range(num_epochs):
        model.train()
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

        # add test loss
        model.eval()
        decoder_y = test_y[:, :-1, :]
        decoder_t = test_y[:, 1:, :]
        decoder_t = decoder_t.reshape(-1)
        decoder_t = torch.tensor(decoder_t, dtype=torch.long)
        scores = model.generate(test_x, decoder_y, decoder_y.shape[1], device)
        test_loss = criterion(scores.reshape(-1, 2), decoder_t)
        test_losses.append(test_loss.item())

    # plot loss curve
    plt.plot(train_losses, label="train", alpha=0.5, c="r")
    plt.plot(test_losses, label="test", alpha=0.5, c="b")
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
