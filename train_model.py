import matplotlib.pyplot as plt
import torch


def train_net(model, criterion, optimizer, num_epochs, train_loader, test_x, test_y, device):

    train_losses = []
    test_losses = []

    for _ in range(num_epochs):
        runnning_loss = 0.0
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

def train_timeseries_net(model, criterion, optimizer, num_epochs,
                         train_loader, test_x, test_y, device):
    train_losses = []
    test_losses = []

    for _ in range(num_epochs):
        runnning_loss = 0.0
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # forward
            score_y = model(batch_x, device)
            loss = criterion(score_y.reshape(-1, 2), batch_y.reshape(-1))

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
        pred_y = model(test_x, device)
        test_loss = criterion(pred_y.reshape(-1, 2), test_y.reshape(-1))
        test_losses.append(test_loss.item())

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


def train_seq2seq_net(model, criterion, optimizer,
                      num_epochs, train_loader, device):
    train_losses = []

    for _ in range(num_epochs):
        runnning_loss = 0.0
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            decoder_yy = batch_y[:, :-1, :]
            decoder_tt = batch_y[:, 1:, :].to(torch.long)

            # forward
            score_y = model(batch_x, decoder_yy, device)
            loss = criterion(score_y.reshape(-1, 2), decoder_tt.reshape(-1))

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
