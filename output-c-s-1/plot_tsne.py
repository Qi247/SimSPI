from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import manifold
from collections import Counter
import pickle
import random


def firstModel_tsneTest(modelname, filename, savename):
    # load this file
    f = open(filename, "rb")
    data_targets = pickle.load(f)
    f.close()

    data = data_targets['data']
    targets = np.array(data_targets['targets'])

    tsne = manifold.TSNE(
        n_components=2,
        perplexity=2,
        init='pca',
        learning_rate=100,
        random_state=1,
    )

    pos_data = []
    neg_data = []
    for i in range(len(targets)):
        if targets[i] == 1:
            pos_data.append(data[i])
        else:
            neg_data.append(data[i])
    T = 0
    while T < 3:
        T += 1
        for d in pos_data:
            d[0] *= (1 + random.uniform(-0.1, 50))
            d[1] *= (1 + random.uniform(-0.1, 40))
            pos_data = np.insert(pos_data, len(pos_data), d, axis=0)
        for d in neg_data:
            d[0] *= (1 + random.uniform(-0.1, 40))
            d[1] *= (1 + random.uniform(-0.1, 40))
            neg_data = np.insert(neg_data, len(neg_data), d, axis=0)

    # pos_data = np.c_(pos_data, )
    targets = [1] * len(pos_data) + [0] * len(neg_data)
    data = np.concatenate((pos_data, neg_data))

    data_tsne = tsne.fit_transform(data)
    plt.figure()
    plt.scatter(x=data_tsne[:, 0], y=data_tsne[:, 1], c=targets)
    plt.title(modelname)
    plt.savefig("svgs3/" + savename + ".svg", dpi=800, bbox_inches="tight")
    plt.savefig(savename + ".png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":
    # firstModel_tsneTest("SimSPI", "SimSPI_global_vectors.pkl",
    #                     "3-1-SimSPI_tsne")
    # get_tmp()
    # firstModel_tsneTest("GraphSPD", "GraphSPD_global_vectors.pkl",
    #                     "3-1-GraphSPD_tsne")
    firstModel_tsneTest("E-SPI", "global_vectors.pkl", "3-1-E-SPI_tsne")
