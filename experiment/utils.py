"""
This module contains several functions that are used in various stages of the process
"""
import json
import operator as op
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, cohen_kappa_score
import pandas as pd
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data: Dict[str, Union[int, float, str]], save_dir: str = "./"):
    with open(os.path.join(save_dir, "results.json"), mode="wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path) -> Dict[str, Union[int, float, str]]:
    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_object(obj, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(input_path: str):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def cal_auc_score(model, data_loader, device):
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)  # クラス確率を計算
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_probs = torch.tensor(all_probs)
    all_targets = torch.tensor(all_targets)

    if len(torch.unique(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_targets, all_probs, multi_class="ovo")
    
    return auc

def cal_acc_score(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return acc

def cal_metrics(model, data_loader, device):
    acc = cal_acc_score(model, data_loader, device)
    auc = cal_auc_score(model, data_loader, device)
    return {"ACC": acc, "AUC": auc}
