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
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
from omegaconf import OmegaConf


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def plot_confusion_matrix(model, data_loader, device, epoch):
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

    all_targets = torch.tensor(all_targets)
    all_preds = torch.tensor(all_preds)

    unique_labels = torch.unique(all_targets)
    unique_labels = unique_labels.cpu().numpy()
    cm = confusion_matrix(all_targets, all_preds, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix for epoch {epoch+1}")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return Image.open(buf)

def concatenate_images(image_list):
    if not image_list:
        return None

    images = [img for img in image_list]
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)
    
    concatenated_image = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    return concatenated_image