import logging
from time import time

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path

import dataset.dataset as dataset
from dataset import TabularDataFrame
from model import get_model, get_criterion, get_optimizer

from .utils import cal_metrics, load_json, set_seed
import torch
from collections import Counter

logger = logging.getLogger(__name__)


class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)

        self.epochs = config.epochs
        self.model_name = config.model.model_name
        self.criterion_name = config.model.criterion_name
        self.optimizer_name = config.model.optimizer_name

        self.exp_config = config.exp
        self.data_config = config.data

        dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(seed=config.seed, **self.data_config)
        
        self.train_loader = dataframe.train_loader
        self.test_loader = dataframe.test_loader

        self.seed = config.seed
        self.init_writer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_writer(self):
        metrics = [
            "epoch",
            "ACC",
            "AUC",
        ]
        self.writer = {m: [] for m in metrics}

    def add_results(self, epoch, scores: dict, time):
        self.writer["epoch"].append(epoch)
        for m in self.writer.keys():
            if m == "epoch":
                continue
            self.writer[m].append(scores[m])

    def run(self):
        score_all = []
        model = get_model(self.model_name)
        criterion = get_criterion(self.criterion_name)
        optimizer = get_optimizer(self.optimizer_name, model)

        best_acc = 0.0
        all_acc = 0.0
        early_stop_count = 0
        for epoch in range(self.epochs):
            start = time()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            score = cal_metrics(model, self.test_loader, self.device)
            score_all.append(score)
            acc = score["ACC"]

            if acc > best_acc:
                best_acc = acc
                early_stop_count = 0
            else:
                early_stop_count += 1
            logger.info(
                f"[{self.model_name} results ({epoch+1} / {self.epochs})] ACC: {score['ACC']:.4f} | AUC: {score['AUC']:.4f}"
            )

            if early_stop_count >= 10:
                print('Early stopping')
                break

            end = time() - start
            self.add_results(epoch, score, end)

        final_score = Counter()
        for item in score_all:
            final_score.update(item)

        logger.info('Best Accuracy: {:.4f}'.format(best_acc))
        logger.info(
                f"[Mean results] ACC: {(final_score['ACC']/self.epochs):.4f} | AUC: {(final_score['AUC']/self.epochs):.4f}"
            )


class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)


