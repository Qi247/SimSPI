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
from dataloader import InputDataset, collate_fn
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from torch.utils.data import DataLoader

K = 3
_ModelJudEpoch_ = 10
_MsgEmbedDim_ = 128
tempPath = "output"
"""
GGNN参数
"out_channels": 200,
"num_layers": 6,
"aggr": "add",
"bias": true
"""

ggnn_args = {"out_channels": 128, "num_layers": 2, "aggr": "add", "bias": True}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global_vectors = torch.Tensor(0, 2)


def _init():
    global global_vectors


def process_global_vectors():
    print(global_vectors)
    np.save("firstModel_globalVectors.npy", global_vectors)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# def test_gnn():
#     dev = "cuda:0"
#     x = [[random.randint(1, 10) for i in range(10)] for j in range(10)]
#     x = torch.tensor(x, device=dev, dtype=torch.float32)
#     edge_index = [[random.randint(0, 9)]*2 for j in range(10)]
#     for e in edge_index:
#         e[1] = (e[0] + 1) % 10
#     edge_index = torch.tensor(edge_index, device=dev, dtype=torch.int64)

#     print(x, edge_index)
#     print(edge_index.transpose(0, 1))
#     data = Data(x=x, edge_index=edge_index.transpose(0, 1))

#     model = Net(ggnn_args, dev)
#     res = model(data)
#     print(res.size())


def GetMsgEmbed(tokenDict, embedSize):
    '''
    Get the pre-trained weights for embedding layer from the dictionary of msg vocabulary.
    :param tokenDict: the dictionary of msg vocabulary.
    {'tk': 0, 'tk': 1, ..., '<pad>': N}
    :param embedSize: the dimension of the embedding vector.
    :return: preWeights - the pre-trained weights for embedding layer.
    [[n, ...], [n, ...], ...]
    '''
    # If the pre
    if os.path.exists(tempPath + '/msgPreWeights.npy'):
        preWeights = np.load(tempPath + '/msgPreWeights.npy')
        return preWeights

    # number of the vocabulary tokens.
    numTokens = len(tokenDict)

    # initialize the pre-trained weights for embedding layer.
    preWeights = np.zeros((numTokens, embedSize))
    for index in range(numTokens):
        preWeights[index] = np.random.normal(size=(embedSize, ))
    print('[INFO] <GetMsgEmbed> Create pre-trained embedding weights with ' +
          str(len(preWeights)) + ' * ' + str(len(preWeights[0])) + ' matrix.')

    # save preWeights.
    np.save(tempPath + '/msgPreWeights.npy', preWeights, allow_pickle=True)
    print(
        '[INFO] <GetMsgEmbed> Save the pre-trained weights of embedding layer to '
        + tempPath + '/msgPreWeights.npy.')

    return preWeights


class MsgRNNv2(nn.Module):
    '''
    MsgRNN : convert a commit message into a predicted label.
    '''

    def __init__(self, preWeights, hiddenSize=32, hiddenLayers=1):
        '''
        define each layer in the network model.
        :param preWeights: tensor pre-trained weights for embedding layer.
        :param hiddenSize: node number in the hidden layer.
        :param hiddenLayers: number of hidden layer.
        '''

        super(MsgRNNv2, self).__init__()
        # parameters.
        class_num = 2
        vocabSize, embedDim = preWeights.size()
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocabSize,
                                      embedding_dim=embedDim,
                                      device=device)
        self.embedding.load_state_dict({'weight': preWeights})
        self.embedding.weight.requires_grad = True
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedDim,
                            hidden_size=hiddenSize,
                            num_layers=hiddenLayers,
                            bidirectional=True)
        # Fully-Connected Layer
        self.fc = nn.Linear(hiddenSize * hiddenLayers * 2, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * diff_length * 1.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        '''

        # x             batch_size * diff_length * 1
        embeds = self.embedding(x)
        # embeds        batch_size * diff_length * embedding_dim
        inputs = embeds.permute(1, 0, 2)
        # inputs        diff_length * batch_size * (embedding_dim + _DiffExtraDim_)
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        # lstm_out      diff_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])],
                                dim=1)
        # feature_map   batch_size * (hidden_size * num_layers * direction_num)
        # TODO:
        final_out = self.fc(feature_map)  # batch_size * class_num
        return final_out
        # return self.softmax(final_out)      # batch_size * class_num


def CosineSimilarity(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


single_dim = 2


class GNNNet(nn.Module):

    class GNNLayer(nn.Module):

        def __init__(self, gated_graph_conv_args: Dict[str, int]):
            super(GNNNet.GNNLayer, self).__init__()
            self.ggcs = [
                GatedGraphConv(**gated_graph_conv_args).to(device)
                for i in range(4)
            ]
            self.gats = [
                GATConv(-1, gated_graph_conv_args['out_channels'],
                        dropout=0.2).to(device) for i in range(4)
            ]
            self.gcns = [
                GCNConv(-1, gated_graph_conv_args['out_channels']).to(device)
                for i in range(4)
            ]
            self.gcn1 = GCNConv(
                gated_graph_conv_args['out_channels'] * 3,
                gated_graph_conv_args['out_channels']).to(device)
            self.Ws = torch.nn.ParameterList()
            for i in range(4):
                self.Ws.append(
                    torch.nn.Parameter(
                        torch.Tensor(ggnn_args["out_channels"],
                                     ggnn_args["out_channels"])).to(device))
                torch.nn.init.xavier_uniform_(
                    self.Ws[-1], gain=nn.init.calculate_gain("relu"))
            self.weight = torch.nn.Parameter(
                torch.rand(ggnn_args['out_channels']))

        def forward(self, x, edge_index, edge_attr):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index, dtype=torch.int64).to(device)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(device)
            # x *= F.softmax(self.weight)
            # x = x / x.sum(dim=1).reshape(-1, 1)

            x_tmp = torch.zeros((x.size()[0], ggnn_args["out_channels"]),
                                dtype=torch.float32).to(device)
            i = 0
            for i in range(4):
                # 1. calculate the node vectors for per graph
                # print(x, edge_index)
                x_t = self.gcns[i](x, edge_index,
                                   edge_attr[:, i + 2].contiguous())
                x_t = F.relu(x_t)
                # x_tmp += x_t
                x_tmp += torch.mm(x_t, self.Ws[i])
                i += 1
                # break
            # x_tmp = torch.cat([t for t in x_t_l], dim=1)
            # x_tmp = self.gcn1(x_tmp, all_edge_index)
            return x_tmp

    def __init__(self, gated_graph_conv_args: Dict[str, int]):
        super(GNNNet, self).__init__()
        self.gnnlayer = self.GNNLayer(gated_graph_conv_args)
        self.g_feture_layer = nn.Linear(ggnn_args["out_channels"],
                                        ggnn_args["out_channels"])  # W(DxD)

        self.sim_matrix_layer = nn.Linear(ggnn_args["out_channels"], K * 2)
        self.sim_matrix_weight = nn.Linear(ggnn_args["out_channels"],
                                           ggnn_args["out_channels"],
                                           bias=False)  # 加不加bias没啥区别

        self.lstm = nn.LSTM(ggnn_args['out_channels'],
                            ggnn_args['out_channels'],
                            2,
                            batch_first=True)

    def forward(self, X):
        # 1. calculate the node vectors for per graph
        # 2. average the vectors for per graph
        # 3. calculate the g_feature for per graph with average vector
        # 4. transpose g_feature vector
        # 5. multiply (node vectors)NxD x Dx1(g_feature) = Nx1 (weight vector)
        # 6. transpose weight vector (Nx1 -> 1xN)
        # 7. multiply weight vector and node vectors (1xN) x (NxD) = 1xD
        # 8. agg vectors for every graph which named graph matrix(K*2 x out_channels).
        # 9. calculate similarity for all graphs(simple implementation). use two for loop

        # g_matrix = torch.zeros(ggnn_args['out_channels'],
        #                        dtype=torch.float32).to(device)
        # g_matrix_list = []  # contain Intermediate results of gnn.
        x_matrix_list = []
        g_r_list = []
        datas = X["graph"]
        commitMsg = X["msg"]
        for data in datas[:K * 2]:  # K*2
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            # edge_indexs[2].extend(edge_indexs[3])
            # del edge_indexs[3]
            # del edge_indexs[2]
            # del edge_indexs[1]

            # 1. calculate the node vectors for per graph
            x_tmp = self.gnnlayer(x, edge_index, edge_attr)
            x_matrix_list.append(x_tmp)

            # replace
            x_tmp_max, _ = x_tmp.max(dim=0)
            x_tmp_mean = x_tmp.mean(dim=0)
            # g_r = torch.cat([x_tmp_max, x_tmp_mean], dim=0)
            g_r = x_tmp_mean
            # g_r = F.dropout(g_r, p=0.5, training=self.training)

            g_r_list.append(g_r)
            # g_matrix -= g_r

            # ---------- 扁平化方式----------------
            # # 2. average the vectors for per graph
            # tmp_mean = torch.unsqueeze(torch.mean(x_tmp, 0),
            #                            0)  # mean for lines
            # # 3. calculate the g_feature for per graph with average vector
            # g_feature = torch.tanh(self.g_feture_layer(tmp_mean))
            # # print("g_feature.size = ", g_feature.size())
            # # 4. transpose g_feature vector
            # g_feature = g_feature.transpose(0, 1)
            # # 5. multiply (node vectors)NxD x Dx1(g_feature) = Nx1 (weight vector)
            # weight_vector = torch.sigmoid(torch.mm(x_tmp, g_feature))
            # # 6. transpose weight vector (Nx1 -> 1xN)
            # weight_vector = weight_vector.transpose(0, 1)
            # # 7. multiply weight vector and node vectors (1xN) x (NxD) = 1xD
            # g_r = torch.mm(weight_vector, x_tmp)

            # # 8. agg vectors for every graph which named graph matrix(K*2 x out_channels).
            # g_r = torch.unsqueeze(x_tmp[-1], dim=0)
            # g_matrix = torch.cat((g_matrix, g_r))
            # ---------- 扁平化方式----------------

        # -----------------------------------------------------
        # g_matrix for a sample
        # g_matrix = F.tanh(g_matrix)
        # -----------------------------------------------------
        g_matrix = torch.cat([t for t in g_r_list], dim=0)
        g_matrix = torch.unsqueeze(g_matrix, dim=0)

        # ======================================================
        # Begin to calculate sim_matrix(New)
        sim_matrix = torch.zeros(len(x_matrix_list),
                                 len(x_matrix_list),
                                 dtype=torch.float32).to(device)
        # print("before:", g_matrix_list[0])
        for i in range(len(x_matrix_list)):
            # g_matrix_list[i] = torch.tanh(g_matrix_list[i])
            x_matrix_list[i] = F.normalize(x_matrix_list[i], p=2, dim=1)
        # print("after:", g_matrix_list[0])
        for i1 in range(len(x_matrix_list)):
            for i2 in range(len(x_matrix_list)):
                # if i1 == i2:
                # continue
                # tmp_res = torch.tensor(0, dtype=torch.float32).to(device)
                # for l1 in range(g_matrix_list[i1].shape[0]):
                #     for l2 in range(g_matrix_list[i2].shape[0]):
                #         tmp_res += torch.mm(
                #             g_matrix_list[i1][l1].reshape(1, -1),
                #             g_matrix_list[i2][l2].reshape(-1,
                #                                           1)).reshape(1)[0]
                #         tmp_res /= torch.norm(g_matrix_list[i1][l1])
                #         tmp_res /= torch.norm(g_matrix_list[i2][l2])
                # sim_matrix[i1][i2] = tmp_res

                tmp = torch.mm(x_matrix_list[i1],
                               x_matrix_list[i2].transpose(0, 1))
                tmp = torch.tanh(tmp)
                # print(tmp)
                tmp = torch.histc(tmp, bins=21, min=-1.0, max=1.0)
                # print("hist: ", tmp)
                tmp = tmp / (len(x_matrix_list[i1]) * len(x_matrix_list[i2]))
                # print("aa:", tmp)
                tmp = torch.matmul(
                    tmp,
                    torch.tensor([
                        -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
                        -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                        1.0
                    ]).to(device))
                # print(tmp.sum())
                # tmp /= torch.norm(g_matrix[i1])
                # tmp /= torch.norm(g_matrix[i2])
                sim_matrix[i1][i2] = torch.sum(tmp)

        sim_matrix = torch.unsqueeze(sim_matrix, dim=0)  # 1 x K*2 x K*2
        return g_matrix, sim_matrix


class Net(nn.Module):

    def __init__(self, gated_graph_conv_args, preWeights, device):
        super(Net, self).__init__()
        self.msgRnn = MsgRNNv2(preWeights=preWeights)

        self.gnnnet = GNNNet(gated_graph_conv_args)

        self.code_linear1 = nn.Linear(ggnn_args["out_channels"] * K * 2, 20)
        # self.bn1 = nn.BatchNorm2d(1)  # just 1 channel
        self.code_fc = nn.Linear(1 * 20, 2)

        self.code_linear2 = nn.Linear(ggnn_args['out_channels'] * K * 2, K * 2)
        self.sim_fc = nn.Linear(K * 2, 2)

        self.criterion = SemiLoss()
        # self.criterion = nn.NLLLoss()
        # self.criterion = nn.CrossEntropyLoss()

    def backnet(self, code_result_tmp, sim_result_tmp):
        code_result = self.code_linear1(
            code_result_tmp)  # N x K*2*20 -> N x 20
        code_result = F.relu(code_result)
        feature_of_G = code_result
        code_result = self.code_fc(code_result)
        code_result = F.log_softmax(code_result, dim=1)

        code_strength = self.code_linear2(code_result_tmp)  # N x K*2
        sim_vector = torch.mean(sim_result_tmp, dim=2)  # N x K*2
        sim_vector = torch.mul(sim_vector, code_strength)  # N x K*2
        sim_result = self.sim_fc(sim_vector)
        sim_result = F.log_softmax(sim_result, dim=1)
        r = code_result * 0.4 + sim_result * 0.6
        return r, feature_of_G

    def forward(self, Xs):
        global global_vectors

        output = torch.empty(0, 2, dtype=torch.float32, device=device)
        code_result_tmp = torch.empty(0,
                                      ggnn_args["out_channels"] * K * 2,
                                      dtype=torch.float32,
                                      device=device)
        sim_result_tmp = torch.empty(0,
                                     K * 2,
                                     K * 2,
                                     dtype=torch.float32,
                                     device=device)
        for X in Xs:
            g_matrix, sim_matrix = self.gnnnet(X)
            code_result_tmp = torch.cat((code_result_tmp, g_matrix), 0)
            sim_result_tmp = torch.cat((sim_result_tmp, sim_matrix), 0)
        r, feature_of_G = self.backnet(code_result_tmp, sim_result_tmp)

        return r, feature_of_G

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DModel(nn.Module):

    def __init__(self):
        super(DModel, self).__init__()
        self.fc1 = nn.Linear(21, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        r = F.log_softmax(x, dim=1)
        return r


class SemiLoss(object):

    def __init__(self):
        self.margin = torch.nn.Parameter(torch.Tensor(1).to(device))
        self.lamda = torch.nn.Parameter(torch.Tensor(1).to(device))
        self.gama = torch.nn.Parameter(torch.Tensor(1).to(device))
        torch.nn.init.constant_(self.margin, 0.1)
        torch.nn.init.constant_(self.lamda, 0.3)
        torch.nn.init.constant_(self.gama, 0.1)

    def __call__(self, outputs_x, targets_x, outputs_u, D_model,
                 feature_of_G_x, feature_of_G_u):
        # 1. calculate the classifier loss
        # Lx = -torch.mean(
        # torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lx = nn.NLLLoss()(outputs_x, targets_x)

        # 2.
        probs_u = torch.log_softmax(outputs_u, dim=1).to(device)
        targets_u = torch.argmax(probs_u, dim=1).to(device)
        D_x = torch.cat((feature_of_G_x, targets_x.view(-1, 1)),
                        dim=1).to(device)
        D_u = torch.cat((feature_of_G_u, targets_u.view(-1, 1)),
                        dim=1).to(device)
        D_data = torch.cat((D_x, D_u), dim=0).to(device)
        D_output = D_model(D_data)
        D_targets = torch.cat(
            (torch.tensor([1] * targets_x.shape[0]),
             torch.tensor([0] * targets_u.shape[0]))).to(device)
        Ld = nn.NLLLoss()(D_output, D_targets)
        # 2. calculate the dicriminace
        # targets_d = torch.tensor([1] * outputs_x.shape[0] +
        #                          [0] * outputs_u.shape[0]).to(device)
        # Ld = nn.CrossEntropyLoss()(outputs_d, targets_d)
        # Ld = - \
        #     torch.mean(torch.sum(F.log_softmax(
        #         outputs_d, dim=1) * targets_d, dim=1))

        # 3. calculate the kl_div for two domain data.
        # probs_u = torch.softmax(outputs_u, dim=1)
        probs_x = outputs_x
        probs_u = torch.log_softmax(outputs_u, dim=1)
        # concat all samples in batch
        probs_all = torch.cat((probs_u, probs_x), dim=0)
        targets_u = torch.argmax(probs_u, dim=1)
        targets_all = torch.cat((targets_x, targets_u),
                                dim=0).reshape(-1, 1).float()
        Lk = F.kl_div(probs_all, targets_all, None, None, 'batchmean')

        # 4. entropy minimization of unlabeled data.
        # Lu = torch.mean(
        #     torch.clamp(
        #         torch.sum(-F.log_softmax(outputs_u, dim=1) *
        #                   F.softmax(outputs_u, dim=1),
        #                   dim=1) - self.margin,  # note:
        #         min=0))
        Lu = torch.mean(
            torch.clamp(
                torch.sum(outputs_x[:, 0] * outputs_x[:, 1]) - 0.4,  # note: 
                min=0))

        # return Lx + Ld + Lk + Lu
        return Lx + 0.05 * Lu + Ld * 0.05


def train(model, D_model, dataloader_x, dataloader_u, optimizer):
    model.train()
    trainloss = 0.0
    trainacc = 0.0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    true_pos_u = 0
    true_neg_u = 0
    false_pos_u = 0
    false_neg_u = 0
    for (data_x, tags_x), (data_u, tags_u) in zip(dataloader_x, dataloader_u):
        # data, tags = data.to(device), tags.to(device)
        tags_x = tags_x.to(device)
        tags_u = tags_u.to(device)

        outputs_x, feature_of_G_x = model(data_x)
        outputs_u, feature_of_G_u = model(data_u)
        loss = model.criterion(outputs_x, tags_x, outputs_u, D_model,
                               feature_of_G_x, feature_of_G_u)
        trainloss += loss.item() * len(data_x)
        # trainacc += (output.argmax(dim=1) == tags).sum().item()
        # ========  Focal Loss Method ===========
        # c = focal_loss()
        # loss = c(output, tags)
        # trainloss += loss.item()
        # output1 = nn.LogSoftmax(dim=0)(output)
        # trainacc += (output1.argmax(dim=1) == tags).sum().item()

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 误差反向传播
        optimizer.step()  # 更新参数

        # 统计true_pos, true_neg, flase_pos, false_neg
        prediction = outputs_x.argmax(dim=1)
        for i in range(len(prediction)):
            if prediction[i] == tags_x[i]:
                if prediction[i] == 0:
                    true_neg += 1
                else:
                    true_pos += 1
            else:
                if prediction[i] == 0:
                    false_neg += 1
                else:
                    false_pos += 1

        prediction = outputs_u.argmax(dim=1)
        for i in range(len(prediction)):
            if prediction[i] == tags_u[i]:
                if prediction[i] == 0:
                    true_neg_u += 1
                else:
                    true_pos_u += 1
            else:
                if prediction[i] == 0:
                    false_neg_u += 1
                else:
                    false_pos_u += 1
    return trainloss / (len(dataloader_x.dataset) +
                        len(dataloader_u.dataset)), [
                            true_pos, true_neg, false_pos, false_neg
                        ], [true_pos_u, true_neg_u, false_pos_u, false_neg_u]


def eval(model, D_model, testloader, u_testloader):
    model.eval()
    testloss = 0.0
    testacc = 0.0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    true_pos_u = 0
    true_neg_u = 0
    false_pos_u = 0
    false_neg_u = 0
    if len(testloader) == 0:
        return 0, [1, 1, 0, 0], [1, 1, 0, 0]
    with torch.no_grad():
        for (data_x, tags_x), (data_u, tags_u) in zip(testloader,
                                                      u_testloader):
            tags_x = tags_x.to(device)
            tags_u = tags_u.to(device)

            outputs_x, feature_of_G_x = model(data_x)
            outputs_u, feature_of_G_u = model(data_u)

            loss = model.criterion(outputs_x, tags_x, outputs_u, D_model,
                                   feature_of_G_x, feature_of_G_u)
            testloss += loss.item() * len(data_x)

            # 统计true_pos, true_neg, flase_pos, false_neg
            prediction = outputs_x.argmax(dim=1)
            for i in range(len(prediction)):
                if prediction[i] == tags_x[i]:
                    if prediction[i] == 0:
                        true_neg += 1
                    else:
                        true_pos += 1
                else:
                    if prediction[i] == 0:
                        false_neg += 1
                    else:
                        false_pos += 1

            prediction = outputs_u.argmax(dim=1)
            for i in range(len(prediction)):
                if prediction[i] == tags_u[i]:
                    if prediction[i] == 0:
                        true_neg_u += 1
                    else:
                        true_pos_u += 1
                else:
                    if prediction[i] == 0:
                        false_neg_u += 1
                    else:
                        false_pos_u += 1
    return testloss / (len(testloader.dataset) + len(u_testloader.dataset)), [
        true_pos, true_neg, false_pos, false_neg
    ], [true_pos_u, true_neg_u, false_pos_u, false_neg_u]


# class SemiLoss(object):

#     def __call__(self, outputs_x, targets_x):
#         Lc = nn.NLLLoss()(outputs_x, targets_x)

#         Lu = torch.mean(
#             torch.clamp(
#                 torch.sum(outputs_x[:, 0] * outputs_x[:, 1]) - 0.4,  # note:
#                 min=0))

#         return Lc + Lu * 0.05

# def train(model, dataloader_x, optimizer):
#     model.train()
#     trainloss = 0.0
#     trainacc = 0.0
#     true_pos = 0
#     true_neg = 0
#     false_pos = 0
#     false_neg = 0
#     for (data_x, tags_x) in dataloader_x:
#         # data, tags = data.to(device), tags.to(device)
#         tags_x = tags_x.to(device)
#         optimizer.zero_grad()  # 梯度清零

#         outputs_x = model(data_x)
#         loss = model.criterion(outputs_x, tags_x)
#         trainloss += loss.item()
#         # trainacc += (output.argmax(dim=1) == tags).sum().item()
#         # ========  Focal Loss Method ===========
#         # c = focal_loss()
#         # loss = c(output, tags)
#         # trainloss += loss.item()
#         # output1 = nn.LogSoftmax(dim=0)(output)
#         # trainacc += (output1.argmax(dim=1) == tags).sum().item()

#         loss.backward()  # 误差反向传播
#         optimizer.step()  # 更新参数

#         # 统计true_pos, true_neg, flase_pos, false_neg
#         prediction = outputs_x.argmax(dim=1)
#         for i in range(len(prediction)):
#             if prediction[i] == tags_x[i]:
#                 if prediction[i] == 0:
#                     true_neg += 1
#                 else:
#                     true_pos += 1
#             else:
#                 if prediction[i] == 0:
#                     false_neg += 1
#                 else:
#                     false_pos += 1
#     return trainloss / len(dataloader_x.dataset), [
#         true_pos, true_neg, false_pos, false_neg
#     ]

# def eval(model, testloader_x):
#     model.eval()
#     testloss = 0.0
#     testacc = 0.0
#     true_pos = 0
#     true_neg = 0
#     false_pos = 0
#     false_neg = 0
#     if len(testloader_x) == 0:
#         return 0, [1, 1, 0, 0]
#     with torch.no_grad():
#         for (data_x, tags_x) in testloader_x:
#             tags_x = tags_x.to(device)
#             outputs_x = model(data_x)
#             loss = model.criterion(outputs_x, tags_x)
#             testloss += loss.item()

#             # 统计true_pos, true_neg, flase_pos, false_neg
#             prediction = outputs_x.argmax(dim=1)
#             for i in range(len(prediction)):
#                 if prediction[i] == tags_x[i]:
#                     if prediction[i] == 0:
#                         true_neg += 1
#                     else:
#                         true_pos += 1
#                 else:
#                     if prediction[i] == 0:
#                         false_neg += 1
#                     else:
#                         false_pos += 1

#     return testloss / len(testloader_x.dataset), [
#         true_pos, true_neg, false_pos, false_neg
#     ]


def plot_loss_acc(loss_list, acc_list, path=["loss", "acc"], epochs=None):
    plt.figure()
    plt.plot(np.arange(1, len(loss_list) + 1), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(1, len(loss_list))  # 右边可以取到
    # plt.xticks(np.arange(1,len(loss_list)+1))
    plt.grid()
    plt.savefig(path[0] + ".png", dpi=800)

    plt.figure()
    plt.plot(np.arange(1, len(acc_list) + 1), acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(1, len(acc_list))  # 右边可以取到
    # plt.ylim(0,1)
    # plt.xticks(np.arange(1,len(acc_list)+1))
    plt.grid()
    plt.savefig(path[1] + ".png", dpi=800)


def cal_matrix(matrix_args):
    # matrix_args: [true_pos, true_neg, false_pos, false_neg]
    print(matrix_args)
    total = matrix_args[0] + matrix_args[1] + matrix_args[2] + matrix_args[3]
    acc = (matrix_args[0] + matrix_args[1]) / total
    pre = matrix_args[0] / (matrix_args[0] + matrix_args[2] + 1)
    recall = matrix_args[0] / (matrix_args[0] + matrix_args[3] + 1)
    f1 = 2 * (pre * recall) / (pre + recall + 0.001)
    fpr = matrix_args[2] / (matrix_args[1] + matrix_args[2] + 1)
    return acc, pre, recall, f1, fpr


testPath = "./testdata2"
target_testPath = "./testdata3"


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
    allCommits = commits

    pos_dataset = []
    neg_dataset = []
    label = 0
    # for root, _, filelist in os.walk(path):
    for commit in allCommits[:400]:
        # for commit in ['10b618827507fbdbe7cf1a9b1f2c81d254dcd8b8']:
        sample = []
        process_count = 0
        for file in sorted(os.listdir(os.path.join(path, commit))):
            if file[-4:] == '.pkl':
                process_count += 1
                if process_count > 10:
                    break
                # read a numpy graph file.
                f = open(os.path.join(os.path.join(path, commit), file), "rb")
                graph = pickle.load(f)
                f.close()
                # graph = np.load(os.path.join(os.path.join(testPath, commit),
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
                sample.append(data)
        # # append the Data instance to dataset.
        if len(sample) == 0:
            continue
        if label == 0:
            neg_dataset.append(sample)
        else:
            pos_dataset.append(sample)

    # print(len(neg_dataset), len(pos_dataset))
    minSize = min(len(neg_dataset), len(pos_dataset))
    minSize = 400
    neg_dataset = neg_dataset[:minSize]
    pos_dataset = pos_dataset[:minSize]

    train_dataset = []
    valid_dataset = []
    train_dataset.extend(pos_dataset[:int(len(pos_dataset) * 0.9)])
    train_dataset.extend(neg_dataset[:int(len(neg_dataset) * 0.9)])
    valid_dataset.extend(pos_dataset[int(len(pos_dataset) * 0.9):])
    valid_dataset.extend(neg_dataset[int(len(neg_dataset) * 0.9):])

    return train_dataset, valid_dataset, files


# dataDir = "./qemu"

if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)
    batch_size = 16
    lr = 0.001
    maxlen = 200  # 传输层payload最大值P
    train_rate = 0.8  # 训练和测试的比例
    if not os.path.exists(tempPath):
        os.makedirs(tempPath)

    ###########################  Data ####################################
    # ------------------------  Create Dataset  ---------------------------
    pwd = os.getcwd()

    train_dataset, valid_dataset, files = GetDataset(path=testPath)
    # print(len(train_dataset))

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

    # target_train_dataloader = train_dataloader
    # target_valid_dataloader = valid_dataloader

    target_train_dataset, target_valid_dataset, target_files = GetDataset(
        path=target_testPath)
    # print(len(train_dataset))

    target_train_dataset = InputDataset(target_train_dataset)
    target_train_dataloader = DataLoader(dataset=target_train_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_fn)

    target_valid_dataset = InputDataset(target_valid_dataset)
    target_valid_dataloader = DataLoader(dataset=target_valid_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_fn)

    print("create datasets success!!!")
    ############################ 模型 #############################################
    f = open("tokenDict.pkl", "rb")
    tokenDict = pickle.load(f)
    msgPreWeights = GetMsgEmbed(tokenDict=tokenDict, embedSize=_MsgEmbedDim_)
    msgPreWeights = torch.from_numpy(msgPreWeights)
    model = Net(ggnn_args, msgPreWeights, device)
    model = model.to(device)
    # model.load_state_dict(torch.load("output/01-16-13-51-0.500000.pkl"))
    # model.load_state_dict(torch.load("10-06-19-48-0.785667.pkl"))
    model.load_state_dict(torch.load("output-4edges/model.pth"))

    D_model = DModel().to(device)

    # ------------------- criterion and optimizer and scheduler  -------------------
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=0.00001)
    # optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)#选择优化器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    ##############################################################################
    ##############################################################################
    # 训练
    loss_list = []
    acc_list = [0.0]
    valid_loss_list = []
    valid_acc_list = [0.0]
    target_valid_acc_list = [0.0]

    num_epoch = 1000
    for epoch in range(num_epoch):
        torch.cuda.empty_cache()
        # train_loss, train_matrix_args, target_train_matrix_args = train(
        #     model, D_model, train_dataloader, target_train_dataloader, optimizer)
        valid_loss, valid_matrix_args, target_valid_matrix_args = eval(
            model, D_model, valid_dataloader, target_valid_dataloader)
        # testloss, testacc = eval(model, test_dataloader, criterion, device)

        # calculate matrix for three types
        # train_acc, train_pre, train_recall, train_f1, train_fpr = cal_matrix(
        #     train_matrix_args)
        valid_acc, valid_pre, valid_recall, valid_f1, valid_fpr = cal_matrix(
            valid_matrix_args)
        target_valid_acc, target_valid_pre, target_valid_recall, target_valid_f1, target_valid_fpr = cal_matrix(
            target_valid_matrix_args)
        # loss_list.append(train_loss)
        # acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        target_valid_acc_list.append(target_valid_acc)

        # print(
        # 'epoch:{:4d}  train_loss:{:10f}    train_acc:{:10f}   train_pre:{:10f}  train_recall{:10f}   train_f1:{:10f}   train_fpr:{:10f}'
        # .format(epoch + 1, train_loss, train_acc, train_pre, train_recall,
        #         train_f1, train_fpr))
        print(
            '             valid_loss:{:10f}    valid_acc:{:10f}   valid_pre:{:10f}  valid_recall{:10f}   valid_f1:{:10f}   valid_fpr:{:10f}'
            .format(valid_loss, valid_acc, valid_pre, valid_recall, valid_f1,
                    valid_fpr))
        print(
            '            u_valid_loss:{:10f}  u_valid_acc:{:10f} u_valid_pre:{:10f} u_valid_recall{:10f} u_valid_f1:{:10f} u_valid_fpr:{:10f}'
            .format(valid_loss, target_valid_acc, target_valid_pre,
                    target_valid_recall, target_valid_f1, target_valid_fpr))
        print(valid_loss_list)
        print(valid_acc_list)
        print(target_valid_acc_list)
        print()

        # save the best model.
        # torch.save(model.state_dict(), tempPath + '/model.pth')
        # if (valid_acc_list[-1] > max(valid_acc_list[0:-1])):
        #     print("=================")
        #     cur = time.localtime(time.time())
        #     torch.save(
        #         model.state_dict(),
        #         tempPath + "/{:02d}-{:02d}-{:02d}-{:02d}-{:8f}.pkl".format(
        #             cur.tm_mon, cur.tm_mday, cur.tm_hour, cur.tm_min,
        #             valid_acc))

        # stop judgement.
        # if (epoch >= _ModelJudEpoch_) and (valid_acc_list[-1] <= min(
        #         valid_acc_list[-1 - _ModelJudEpoch_:-1])):
        #     break
        # 保存准确率和损失列表
        # cur = time.localtime(time.time())
        # with open(
        #         tempPath + "/{:02d}-{:02d}-{:02d}-{:02d}-loss-acc.pkl".format(
        #             cur.tm_mon, cur.tm_mday, cur.tm_hour, cur.tm_min),
        #         "wb") as f:
        #     pickle.dump({"loss": valid_loss_list, "acc": valid_acc_list}, f)
        # # 保存准确率和损失曲线图
        # plot_loss_acc(loss_list=loss_list,
        #               acc_list=acc_list,
        #               path=[tempPath + "/train_loss", tempPath + "/train_acc"])
        # plot_loss_acc(loss_list=valid_loss_list,
        #               acc_list=valid_acc_list,
        #               path=[tempPath + "/valid_loss", tempPath + "/valid_acc"])
