import matplotlib.pyplot as plt


def train_net(model, criterion, optimizer, num_epochs, train_loader, X_test, Y_test, device):

    train_losses = []
    test_losses = []

    for _ in range(num_epochs):
        runnning_loss = 0.0
        for idx, (xx, yy) in enumerate(train_loader):
            xx = xx.to(device)
            yy = yy.to(device)

            # forward
            Y_score = model(xx)
            loss = criterion(Y_score, yy)
            
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
        Y_pred = model(X_test)
        test_loss = criterion(Y_pred, Y_test)
        test_losses.append(test_loss.item())

    # plot loss curve
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")

    return model
