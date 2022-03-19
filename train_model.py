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
