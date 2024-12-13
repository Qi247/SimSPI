from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import GCNConv
import random
from typing import Dict
from torch_geometric.data import Data
import os
import pickle
from dataloader import InputDataset, collate_fn
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from torch.utils.data import DataLoader
import model_train

testPath = "../testdata1"
tokenDictPath = "../tokenDict.pkl"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# get dataset
def GetDataset(path=None):
    '''
    Get the dataset from numpy data files.
    :param path: the path used to store numpy dataset.
    :return: dataset - list of torch_geometric.data.Data
    '''

    # check.
    if None == path:
        print(
            '[Error] <GetDataset> The method is missing an argument \'path\'!')
        return [], []

    # contruct the dataset.
    files = []
    poss = []
    negs = []
    commits = os.listdir(testPath)
    for commit in commits:
        label_file = "../../mygraph/data/qemu/labels/{:s}".format(commit)
        with open(label_file) as f:
            label = int(f.read()[0])
            if label == 0:
                negs.append(commit)
            else:
                poss.append(commit)

    allCommits = []
    for neg in negs[:]:
        allCommits.append(neg)
    for pos in poss[:]:
        allCommits.append(pos)

    pos_dataset = []
    neg_dataset = []
    label = 0
    # for root, _, filelist in os.walk(path):
    for commit in allCommits:
        # for commit in ['10b618827507fbdbe7cf1a9b1f2c81d254dcd8b8']:
        sample = []
        process_count = 0
        for file in sorted(os.listdir(os.path.join(testPath, commit))):
            if file[-4:] == '.pkl':
                process_count += 1
                if process_count > 10:
                    break
                # read a numpy graph file.
                f = open(os.path.join(os.path.join(testPath, commit), file),
                         "rb")
                graph = pickle.load(f)
                f.close()
                # graph = np.load(os.path.join(os.path.join(testPath, commit),
                #                              file),
                #                 allow_pickle=True)
                files.append(
                    os.path.join(os.path.join(testPath, commit), file[:-7]))
                # sparse each element.
                edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
                nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
                edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)
                label = torch.tensor(graph['label'], dtype=torch.long)
                # construct an instance of torch_geometric.data.Data.
                data = Data(edge_index=edgeIndex,
                            x=nodeAttr,
                            edge_attr=edgeAttr,
                            y=label)
                sample.append(data)
        # # append the Data instance to dataset.
        if len(sample) == 0:
            continue
        if label == 0:
            neg_dataset.append(sample)
        else:
            pos_dataset.append(sample)

    minSize = min(len(neg_dataset), len(pos_dataset))
    # minSize = 400
    neg_dataset = neg_dataset[:minSize]
    pos_dataset = pos_dataset[:minSize]

    train_dataset = []
    valid_dataset = []
    train_dataset.extend(pos_dataset[:int(len(pos_dataset) * 0.9)])
    train_dataset.extend(neg_dataset[:int(len(neg_dataset) * 0.9)])
    valid_dataset.extend(pos_dataset[int(len(pos_dataset) * 0.9):])
    valid_dataset.extend(neg_dataset[int(len(neg_dataset) * 0.9):])
    valid_dataset = neg_dataset + pos_dataset

    return train_dataset, valid_dataset, files


def eval_get_res(model, testloader):
    model.eval()
    global_vectors = torch.Tensor(0, 2).to(model_train.device)
    targets = torch.tensor([]).to(model_train.device)

    with torch.no_grad():
        for (data_x, tags_x) in testloader:
            tags_x = tags_x.to(model_train.device)
            outputs_x = model(data_x)
            global_vectors = torch.cat((global_vectors, outputs_x), dim=0)
            targets = torch.cat((targets, tags_x), dim=0)
    with open("global_vectors.pkl", "wb") as f:
        pickle.dump(
            {
                "data": global_vectors.cpu().numpy(),
                "targets": targets.cpu().numpy()
            }, f)


if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)
    batch_size = 16
    lr = 0.0005
    maxlen = 200  # 传输层payload最大值P

    ###########################  Data ####################################
    # ------------------------  Create Dataset  ---------------------------
    pwd = os.getcwd()

    train_dataset, valid_dataset, files = GetDataset(path=testPath)

    train_dataset = InputDataset(train_dataset)
    # train_dataloader = train_dataset.get_loader(
    #     batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)

    valid_dataset = InputDataset(valid_dataset)
    # train_dataloader = train_dataset.get_loader(
    #     batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)

    print("create datasets success!!!")
    ############################ 模型 #############################################
    msgPreWeights = torch.from_numpy(np.load("msgPreWeights.npy"))
    model = model_train.Net(model_train.ggnn_args, msgPreWeights,
                            model_train.device)
    model = model.to(model_train.device)
    # model.load_state_dict(torch.load("03-15-15-53-0.739362.pkl"))
    model.load_state_dict(torch.load("03-15-15-40-0.865248.pkl"))

    ################################## eval_get_res ##############################
    # eval_get_res(model, valid_dataloader)
    # exit(0)

    ################################### eval #####################################

    torch.cuda.empty_cache()

    valid_loss, valid_matrix_args = model_train.eval(model, valid_dataloader)
    # testloss, testacc = eval(model, test_dataloader, criterion, device)

    # calculate matrix for three types
    valid_acc, valid_pre, valid_recall, valid_f1, valid_fpr = model_train.cal_matrix(
        valid_matrix_args)

    print(
        '             valid_loss:{:10f}    valid_acc:{:10f}   valid_pre:{:10f}  valid_recall{:10f}   valid_f1:{:10f}   valid_fpr:{:10f}'
        .format(valid_loss, valid_acc, valid_pre, valid_recall, valid_f1,
                valid_fpr))
    print()
