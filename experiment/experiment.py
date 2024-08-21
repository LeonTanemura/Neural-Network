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

    def init_writer(self):
        metrics = [
            "fold",
            "ACC",
            "AUC",
        ]
        self.writer = {m: [] for m in metrics}

    def add_results(self, i_fold, scores: dict, time):
        self.writer["fold"].append(i_fold)
        for m in self.writer.keys():
            if m == "fold":
                continue
            self.writer[m].append(scores[m])

    def run(self):
        model = get_model(self.model_name)
        criterion = get_criterion(self.criterion_name)
        optimizer = get_optimizer(self.optimizer_name, model)

        best_acc = 0.0
        all_acc = 0.0
        early_stop_count = 0
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Test the network
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    output = model(data)
                    prediction = output.argmax(dim=1, keepdim=True)
                    correct += prediction.eq(target.view_as(prediction)).sum().item()
                    total += target.size(0)

            acc = 100. * correct / total
            all_acc += acc
            if acc > best_acc:
                best_acc = acc
                early_stop_count = 0
            else:
                early_stop_count += 1
            logger.info('Epoch: {} Accuracy: {}/{} ({:.2f}%)'.format(epoch, correct, total, acc))

            if early_stop_count >= 10:
                print('Early stopping')
                break

        logger.info('Best Accuracy: {:.2f}%'.format(best_acc))
        logger.info('Mean Accuracy: {:.2f}%'.format(all_acc / self.epochs))

        # logger.info(
        #         f"[{self.model_name} results] MSE: {(final_score['MSE']/self.n_splits)} | MAE: {(final_score['MAE']/self.n_splits)} | "
        #         f"RMSE: {(final_score['RMSE']/self.n_splits)} | QWK: {(final_score['QWK']/self.n_splits)}"
        #     )


class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)


