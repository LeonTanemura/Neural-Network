from experiment.utils import set_seed

from .model import Net
import torch
import torch.nn as nn
import torch.optim as optim

def get_model(name):
    if name == "net":
        return Net()
    else:
        raise KeyError(f"{name} is not defined.")

def get_criterion(name):
    if name == "CrossEntropy":
        return nn.CrossEntropyLoss()
    else:
        raise KeyError(f"{name} is not defined.")

def get_optimizer(name, model):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    elif name == "Adam":
        return optim.Adam(model.parameters(), lr=0.001)
    elif name == "AdamW":
        return optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    elif name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    elif name == "Adadelta":
        return optim.Adadelta(model.parameters(), lr=1.0)
    else:
        raise KeyError(f"{name} is not defined.")

