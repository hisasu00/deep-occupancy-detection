from functools import partial
import os
import pickle

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import WandbLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import AttentionRNN
from train_model import train_timeseries_net


def pipeline_rnn(dataset, options, config, trainer=train_timeseries_net):
    # preprocessing
    train_loader, val_x, val_y, test_x, test_y = preprocess_for_rnn(dataset, options["device"])
    options["dataset"]["train_loader"] = train_loader
    options["dataset"]["val_x"] = val_x
    options["dataset"]["val_y"] = val_y
    
    # training and params optimization
    hyperopt = HyperOptSearch(metric="loss", mode="min")
    scheduler = ASHAScheduler(
        metric='loss', mode='min', max_t=1000,
        grace_period=12, reduction_factor=2)

    analysis = tune.run(
        partial(trainer, options=options),
        config=config,
        num_samples=2,
        search_alg=hyperopt,
        resources_per_trial={'cpu':4, 'gpu':1},
        scheduler=scheduler,
        loggers=[WandbLogger]
    )

    # test
    model = standby_rnn_for_test(analysis, options)
    acc = test(model, test_x, test_y, options["device"])
    return acc


def preprocess_for_rnn(dataset, device):
    train_x, train_y, val_x, val_y, test_x, test_y = dataset.train_x, dataset.train_y, dataset.val_x,\
                                                     dataset.val_y, dataset.test_x, dataset.test_y

    train_num_days = train_x.shape[0]
    val_num_days = val_x.shape[0]
    test_num_days = test_x.shape[0]

    train_x = train_x.reshape(train_num_days * sequence_length, feature_size)
    val_x = val_x.reshape(val_num_days * sequence_length, feature_size)
    test_x = test_x.reshape(test_num_days * sequence_length, feature_size)

    ss = preprocessing.StandardScaler()
    ss.fit(train_x[:, :continuous_feature_size])

    train_x[:, :continuous_feature_size] = \
        ss.transform(train_x[:, :continuous_feature_size])

    val_x[:, :continuous_feature_size] = \
        ss.transform(val_x[:, :continuous_feature_size])

    test_x[:, :continuous_feature_size] = \
        ss.transform(test_x[:, :continuous_feature_size])

    train_x = train_x.reshape(train_num_days, sequence_length, feature_size)
    val_x = val_x.reshape(val_num_days, sequence_length, feature_size)
    test_x = test_x.reshape(test_num_days, sequence_length, feature_size)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True)
    
    return train_loader, val_x, val_y, test_x, test_y


def standby_rnn_for_test(analysis, options):
    best = analysis.get_best_config(metric="loss", mode="min")
    logdir = analysis.get_best_logdir("loss", mode="min")
    state_dict = torch.load(os.path.join(logdir, "rnn.pth"))

    model = AttentionRNN(input_size=options["params"]["input_size"], hidden_size=best["hidden_size"],
                         num_layers=options["params"]["num_layers"], num_classes=options["params"]["num_classes"],
                         fc_sizes=[best["fc_size_0"], best["fc_size_1"], best["fc_size_2"]]).to(options["device"])
    model.load_state_dict(state_dict)
    return model


def test(model, test_x, test_y, device):
    model.eval()
    score_y = model(test_x, device).reshape(-1)
    score_y = torch.sigmoid(score_y)
    pred_y = torch.tensor([1 if i > 0.5 else 0 for i in score_y]).to(device)
    test_y = test_y.reshape(-1)
    test_y = test_y.to('cpu').detach().numpy().copy()
    pred_y = pred_y.to('cpu').detach().numpy().copy()
    acc = sum(pred_y == test_y) / int(test_y.shape[0])
    return acc


def pipeline_benchmarks(dataset, config, options, trainer, name):
    # preprocessing
    train_x, train_y, val_x, val_y, test_x, test_y = dataset.train_x, dataset.train_y, dataset.val_x,\
                                                     dataset.val_y, dataset.test_x, dataset.test_y

    train_x = train_x.reshape(-1, train_x.shape[2])
    train_y = train_y.reshape(-1)
    test_x = test_x.reshape(-1, test_x.shape[2])
    test_y = test_y.reshape(-1)
    val_x = val_x.reshape(-1, val_x.shape[2])
    val_y = val_y.reshape(-1)
    options["dataset"]["train_x"] = train_x
    options["dataset"]["train_y"] = train_y
    options["dataset"]["val_x"] = val_x
    options["dataset"]["val_y"] = val_y
    
    # training and params optimization
    hyperopt = HyperOptSearch(metric="accuracy", mode="max")
    analysis = tune.run(
        partial(trainer, options=options),
        config=config,
        num_samples=2,
        search_alg=hyperopt,
        resources_per_trial={'cpu':4, 'gpu':1}
    )   
    
    # test
    model = standby_benchmarks_for_test(analysis, name)
    pred_y = model.predict(test_x)
    acc = sum(pred_y == test_y) / int(test_y.shape[0])
    return acc


def train_rf(config, options):
    train_x, train_y, val_x, val_y = options["dataset"].values()
    rf = RandomForestClassifier(max_depth=config["max_depth"],min_samples_leaf=config["min_samples_leaf"],
                                min_samples_split=config["min_samples_split"], n_estimators=config["n_estimators"])
    rf.fit(train_x, train_y)
    
    pred_y = rf.predict(val_x)
    acc = sum(val_y == pred_y) / pred_y.shape[0]
    tune.report(accuracy=acc)
    
    with open(f"./rf.pkl", "wb") as out_file:
        pickle.dump(rf, out_file)
        
        
def train_linear(config, options):
    train_x, train_y, val_x, val_y = options["dataset"].values()
    linear = make_pipeline(preprocessing.StandardScaler(), SVC(kernel="linear", C = config["C"]))
    linear.fit(train_x, train_y)
    
    pred_y = linear.predict(val_x)
    acc = sum(val_y == pred_y) / pred_y.shape[0]
    tune.report(accuracy=acc)
    
    with open(f"./linear.pkl", "wb") as out_file:
        pickle.dump(linear, out_file)

        
def train_rbf(config, options):
    train_x, train_y, val_x, val_y = options["dataset"].values()
    linear = make_pipeline(preprocessing.StandardScaler(), SVC(kernel="rbf", C = config["C"], gamma=config["gamma"]))
    linear.fit(train_x, train_y)
    
    pred_y = linear.predict(val_x)
    acc = sum(val_y == pred_y) / pred_y.shape[0]
    tune.report(accuracy=acc)
    
    with open(f"./rbf.pkl", "wb") as out_file:
        pickle.dump(linear, out_file)

        
def standby_benchmarks_for_test(analysis, name):
    logdir = analysis.get_best_logdir("accuracy", mode="min")
    with open(os.path.join(logdir, name), 'rb') as f:
        model = pickle.load(f)
    return model


class Dataset(object):
  def __init__(self, train_x, train_y, val_x, val_y, test_x, test_y):
    self.train_x = train_x
    self.train_y = train_y
    self.val_x = val_x
    self.val_y = val_y
    self.test_x = test_x
    self.test_y = test_y


if __name__ == "__main__":
    # Import Module
    import datetime

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, KFold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    x = pd.read_csv("./data/2_X_train.csv").values
    y = pd.read_csv("./data/2_Y_train.csv").values.reshape(-1)

    sequence_length = 16
    num_days = int(x.shape[0] / sequence_length)
    feature_size = x.shape[1]
    continuous_feature_size = 8

    x = x.reshape(num_days, sequence_length, feature_size)
    y = y.reshape(num_days, sequence_length, 1)
    
    # Set Config and Options for Raytune
    rnn_config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_size": tune.choice([25, 50, 100, 150]),
        "weight_decay": tune.choice([0, 1e-7, 1e-5, 1e-3, 1e-1]),
        "eps": tune.choice([1e-11, 1e-8, 1e-5, 1e-3, 1e-1]),
        "fc_size_0": tune.choice([50, 75, 100]),
        "fc_size_1": tune.choice([15, 25, 35]),
        "fc_size_2": tune.choice([5, 10, 20]),
        "wandb": {
            "project": "test",
            "api_key_file": "./wandb_api_key.txt"
        }
    }

    rf_config = {
        'max_depth': tune.choice([5, 10, 15, 20, 50, None]),
        'min_samples_leaf': tune.choice([3, 4, 5]),
        'min_samples_split': tune.choice([2, 8, 10, 12]),
        'n_estimators': tune.choice([50, 100, 200])
    }

    linear_config = {
        'C': tune.choice([0.1, 0.5, 1, 5, 10, 50, 100]),
    }

    rbf_config = {
        'C': tune.choice([0.1, 0.5, 1, 5, 10, 50, 100]),
        "gamma": tune.loguniform(0.0001, 1)
    }

    rnn_options = {
        'dataset': {'train_loader': None, 'val_x': None, 'val_y': None},
        "num_epochs": 100,
        "params": {"input_size": 13, "num_layers": 1, "num_classes": 1},
        "device": device
    }

    benchmarks_options = {
        'dataset': {'train_x': None, 'train_y': None,
                    'val_x': None, 'val_y': None},
    }

    # Raytune
    n_splits = 2
    val_size = 0.1
    num_repeats = 1

    accs = []
    rf_accs = []
    linear_accs = []
    rbf_accs = []

    for repeat in range(num_repeats):
        kf = KFold(n_splits=n_splits, random_state=repeat, shuffle=True)
        running_acc = 0
        rf_running_acc = 0
        linear_running_acc = 0
        rbf_running_acc = 0

        for train_idx, test_idx in kf.split(x):
            # Split Data Into Train, Val, Test
            train_x, test_x = x[train_idx], x[test_idx]
            train_y, test_y = y[train_idx], y[test_idx]
            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_size, train_size=(1-val_size))
            dataset = Dataset(train_x, train_y, val_x, val_y, test_x, test_y)

            # BiLSTM-Attention
            running_acc += pipeline_rnn(dataset, rnn_options, rnn_config, train_timeseries_net)

            # Benchmarks
            rf_running_acc += pipeline_benchmarks(dataset, rf_config, benchmarks_options, train_rf, "rf.pkl")
            linear_running_acc += pipeline_benchmarks(dataset, linear_config, benchmarks_options, train_linear, "linear.pkl")
            rbf_running_acc += pipeline_benchmarks(dataset, rbf_config, benchmarks_options, train_rbf, "rbf.pkl")
            
        # Get Mean Score Among k-fold
        mean_acc = running_acc/n_splits
        rf_mean_acc = rf_running_acc/n_splits
        linear_mean_acc = linear_running_acc/n_splits
        rbf_mean_acc = rbf_running_acc/n_splits

        accs.append(mean_acc)
        rf_accs.append(rf_mean_acc)
        linear_accs.append(linear_mean_acc)
        rbf_accs.append(rbf_mean_acc)
    
    # Logging
    date = datetime.datetime.now()
    date = date.strftime('%Y-%m-%d-%H-%M-%S')

    with open(f'log_{date}.txt', 'w') as f:
        print(f"date: {date}", file=f)
        print(f"BiLSTM-Attention:{np.mean(accs)}({np.std(accs)})", file=f)
        print(f"RF:{np.mean(rf_accs)}({np.std(accs)})", file=f)
        print(f"Linear-SVM:{np.mean(linear_accs)}({np.std(accs)})", file=f)
        print(f"Rbf-SVM:{np.mean(rbf_accs)}({np.std(accs)})", file=f)
