import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

class TabularDataFrame(object):

    def __init__(
        self,
        seed,
        **kwargs,
    ) -> None:
        self.seed = seed
        self.batch_size = kwargs.get('batch_size')
        print(f"batch_size: {self.batch_size}")


class V0(TabularDataFrame):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(root='.', train=True, transform=self.transform, download=True)
        self.test_dataset = datasets.MNIST(root='.', train=False, transform=self.transform, download=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)