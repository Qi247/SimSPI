import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
from torch_geometric.data import Data

K = 3


class InputDataset(TorchDataset):

    def __init__(self, samples):
        """
        samples: [[Data, Data, ...], [Data, Data, ...]]
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # read pkl from path
        X = {}
        data = self.samples[index]
        y = data[0].y

        # now, data is [Data, Data, ...]
        for i in range(K * 2 - len(data)):
            data.append(
                Data(edge_index=[[0], [0]],
                     x=[[0] * 20],
                     edge_attr=[[0, 0, 0, 0, 0, 0]],
                     y=0))
        X["graph"] = data

        # ---------------------------
        # get msgMapping from file
        # msg_path = abs_path.replace("positive_pkl", "positive_msg").replace(
        # "negative_pkl", "negative_msg") + ".npy"
        # msg_mapping = np.load(msg_path)
        X["msg"] = ""
        return X, y

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)


def collate_fn(samples):
    """
    samples: [[X, 0/1], ]
    """
    graphs = []
    msgs = []
    y = []
    X = []
    for data, label in samples:
        graphs.append(data["graph"])
        msgs.append(data["msg"])
        y.append(label)
        X.append(data)
    return X, torch.tensor(y)
