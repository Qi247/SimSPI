import os
import sys
import time
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from libs.nets.PGCN_noAST import PGCN, PGCNTest, PGCNTrain

from preproc import extract_graphs, construct_graphs

import pickle

logsPath = './logs/'
testPath1 = './testdata1/'
testPath2 = './testdata2/'

mdlsPath = './models/'

# parameters
_CLANG_ = 1
_NETXARCHT_ = 'PGCN'
_BATCHSIZE_ = 128
dim_features = 20
start_time = time.time()  #mark start time


class Logger(object):

    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def RunTime():
    pTime = ' [TIME: ' + str(round((time.time() - start_time), 2)) + ' sec]'
    return pTime


# extract graphs
def extractgraphs():
    cnt = 0
    for root, ds, fs in sorted(os.walk(testPath1)):
        process_count = 0
        fs = sorted(fs)
        for file in fs:
            if ('.log' == file[-4:]):
                process_count += 1
                if process_count > 5:
                    break
                filename = os.path.join(root, file).replace('\\', '/')
                savename = (filename + '_mid.npz').replace(
                    testPath1, testPath2)
                if not os.path.exists(os.path.dirname(savename)):
                    os.makedirs(os.path.dirname(savename))
                print(savename)
                cnt += 1
                nodes, edges, nodes0, edges0, nodes1, edges1 = extract_graphs.ReadFile(
                    filename)
                if _CLANG_:
                    nodes = extract_graphs.ProcNodes(nodes, 'PatchCPG')
                    nodes0 = extract_graphs.ProcNodes(nodes0, 'PreCPG')
                    nodes1 = extract_graphs.ProcNodes(nodes1, 'PostCPG')
                label = [0]
                # print(filename)
                commit_id = filename.split("/")[2]
                label_file = "../mygraph/data/qemu/labels/{:s}".format(
                    commit_id)
                with open(label_file) as f:
                    label[0] = int(f.read()[0])
                    # print(label[0])

                np.savez(savename,
                         nodes=nodes,
                         edges=edges,
                         nodes0=nodes0,
                         edges0=edges0,
                         nodes1=nodes1,
                         edges1=edges1,
                         label=label,
                         dtype=object)
                print(
                    f'[INFO] <main> save the graph information into numpy file: [{str(cnt)}] '
                    + savename + RunTime())
                print('=====================================================')
    return


# construct graphs
def constructgraphs():
    cnt = 0
    for root, ds, fs in os.walk(testPath2):
        fs = sorted(fs)
        cnt += 1
        func_idx = -1
        for file in fs:
            if ('_mid.npz' == file[-8:]):
                func_idx += 1
                if func_idx >= 5:
                    break
                filename = os.path.join(root, file).replace('\\', '/')
                savename0 = filename.replace('_mid.npz',
                                             '_{:d}_0.pkl'.format(func_idx))
                savename1 = filename.replace('_mid.npz',
                                             '_{:d}_1.pkl'.format(func_idx))
                print('[INFO] <main> Process the graph numpy file: [' +
                      str(cnt) + '] ' + filename + RunTime())
                nodes, edges, nodes0, edges0, nodes1, edges1, label = construct_graphs.ReadFile(
                    filename)
                nodeDict0, edgeIndex0, edgeAttr0 = construct_graphs.ProcEdges(
                    edges0)
                nodeAttr0, nodeInvalid0 = construct_graphs.ProcNodes(
                    nodes0, nodeDict0)
                nodeDict1, edgeIndex1, edgeAttr1 = construct_graphs.ProcEdges(
                    edges1)
                nodeAttr1, nodeInvalid = construct_graphs.ProcNodes(
                    nodes1, nodeDict1)

                with open(savename0, "wb") as f:
                    pickle.dump(
                        {
                            'edgeIndex': edgeIndex0,
                            'edgeAttr': edgeAttr0,
                            'nodeAttr': nodeAttr0,
                            'label': label,
                            'nodeDict': nodeDict0
                        }, f)
                with open(savename1, "wb") as f:
                    pickle.dump(
                        {
                            'edgeIndex': edgeIndex1,
                            'edgeAttr': edgeAttr1,
                            'nodeAttr': nodeAttr1,
                            'label': label,
                            'nodeDict': nodeDict1
                        }, f)

                # np.savez(savename,
                #          edgeIndex=edgeIndex,
                #          nodeAttr=nodeAttr,
                #          label=label,
                #          nodeDict=nodeDict)
                print(
                    '[INFO] <main> save the graph information into numpy file: ['
                    + str(cnt) + '] ' + savename1 + RunTime())
                print('-----------------------------------------------------')
        # break
    return


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
    commits = os.listdir(path)
    for commit in commits:
        label_file = "../mygraph/data/qemu/labels/{:s}".format(commit)
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
        sample = []
        for file in os.listdir(os.path.join(path, commit)):
            if file[-4:] == '.pkl':
                # read a numpy graph file.
                f = open(os.path.join(os.path.join(path, commit), file), "rb")
                graph = pickle.load(f)
                f.close()
                # graph = np.load(os.path.join(os.path.join(path, commit),
                #                              file),
                #                 allow_pickle=True)
                files.append(
                    os.path.join(os.path.join(path, commit), file[:-7]))
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
                if label == 0:
                    neg_dataset.append(sample)
                else:
                    pos_dataset.append(sample)
                break
                # sample.append(data)
        # # append the Data instance to dataset.
        # if len(sample) == 0:
        #     continue
        # if label == 0:
        #     neg_dataset.append(sample)
        # else:
        #     pos_dataset.append(sample)

    print(len(neg_dataset), len(pos_dataset))
    minSize = max(len(neg_dataset), len(pos_dataset))
    # minSize = 400
    neg_dataset = neg_dataset[:minSize]
    pos_dataset = pos_dataset[:minSize]

    train_dataset = []
    valid_dataset = []
    train_dataset.extend(pos_dataset[:int(len(pos_dataset) * 0.9)])
    train_dataset.extend(neg_dataset[:int(len(neg_dataset) * 0.9)])
    valid_dataset.extend(pos_dataset[int(len(pos_dataset) * 0.9):])
    valid_dataset.extend(neg_dataset[int(len(neg_dataset) * 0.9):])
    valid_dataset = train_dataset = neg_dataset + pos_dataset

    return train_dataset, valid_dataset, files


def cal_matrix(matrix_args):
    # matrix_args: [true_pos, true_neg, false_pos, false_neg]
    # print(matrix_args)
    total = matrix_args[0] + matrix_args[1] + matrix_args[2] + matrix_args[3]
    acc = (matrix_args[0] + matrix_args[1]) / total
    pre = matrix_args[0] / (matrix_args[0] + matrix_args[2] + 1)
    recall = matrix_args[0] / (matrix_args[0] + matrix_args[3] + 1)
    f1 = 2 * (pre * recall) / (pre + recall + 0.001)
    fpr = matrix_args[2] / (matrix_args[1] + matrix_args[2] + 1)
    return acc, pre, recall, f1, fpr


# main
def main():
    model = PGCN(num_node_features=dim_features)
    # model.load_state_dict(
    # torch.load(mdlsPath + f'/model_{_NETXARCHT_}_{dim_features}_10.pth'))
    # model.load_state_dict(torch.load("2024-01-16-15.pth"))

    train_dataset, valid_dataset, files = GetDataset(path=testPath2)
    # print(train_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=_BATCHSIZE_,
                                  shuffle=False)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=_BATCHSIZE_,
                                  shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0002,
                                 weight_decay=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(1000):
        model, lossTrain, dis_array = PGCNTrain(model, train_dataloader,
                                                optimizer, criterion)
        acc_train, _, _, _, _ = cal_matrix(dis_array)

        tp, tn, fp, fn = PGCNTest(model, valid_dataloader)

        acc, pre, recall, f1, fpr = cal_matrix([tp, tn, fp, fn])

        print(i, dis_array, lossTrain, acc_train)
        print(" [", tp, tn, fp, fn, "]", acc)
    # torch.save(model.state_dict(), "2024-01-16-15.pth")

    # filename = logsPath + '/test_results.txt'
    # fp = open(filename, 'w')
    # fp.write(f'filename,prediction\n')
    # for i in range(len(files)):
    #     fp.write(f'{files[i]},{testPred[i]}\n')
    # fp.close()

    return


if __name__ == '__main__':
    logfile = 'test.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # --------------------------------------------------
    # extractgraphs()
    # # check [mid error]
    # constructgraphs()
    # # check [np error]
    # main()
