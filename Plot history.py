# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:38:52 2022

@author: teun_
"""

import matplotlib.pyplot as plt
import numpy as np
import os

#%% Set working directory

os.chdir("C:/Users/teun_/OneDrive/Documenten/Master Data Science and Society/Thesis")
datadir_inter = os.getcwd().replace("\\", "/") + "/Data/Inter-dataset/"
cleandataDIR_inter = os.getcwd().replace("\\", "/") + "/Data/Clean/Inter-datasets/"
cleandataDIR_intra = os.getcwd().replace("\\", "/") + "/Data/Clean/Intra-datasets/"
modelDIR = "C:/Users/teun_/OneDrive/Documenten/Master Data Science and Society/Thesis/Models/"

#%% Load train and validation performance of CNN

datasets = ['zheng', 'baron_human', 'amb']

cnn_performance = {}

for dataset in datasets:
    cnn_performance[dataset] = np.load(cleandataDIR_intra + 'relu_history_' + dataset + '_cnn.npy', allow_pickle=True).item()
    print("Max val_accuracy", dataset, ": ", max(cnn_performance[dataset]["val_accuracy"]))

datasets = ['inter_pbmc']

for dataset in datasets:
    cnn_performance[dataset] = np.load(cleandataDIR_inter + 'relu_history_' + dataset + '_cnn.npy', allow_pickle=True).item()
    print("Max val_accuracy", dataset, ": ", max(cnn_performance[dataset]["val_accuracy"]))

#%% Plot train and validation performance CNN

datasets = ['zheng', 'baron_human', 'amb', 'inter_pbmc']

for dataset in datasets:
    plt.plot(cnn_performance[dataset]["val_accuracy"])

plt.title('CNN accuracy')
plt.xlim([0.0, 105])
plt.ylim([0.0, 1.05])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(datasets, loc='upper left')
plt.savefig('Figures/History CNN.png', dpi=300)
plt.show()





#%%#########################
##### LSTM performance #####
############################

# Load train and validation performance of LSTM

datasets = ['zheng', 'baron_human', 'amb']

LSTM_performance = {}

for dataset in datasets:
    LSTM_performance[dataset] = np.load(cleandataDIR_intra + 'tanh_history_' + dataset + '_LSTM.npy', allow_pickle=True).item()
    print("Max val_accuracy", dataset, ": ", max(LSTM_performance[dataset]["val_accuracy"]))

datasets = ['inter_pbmc']

for dataset in datasets:
    LSTM_performance[dataset] = np.load(cleandataDIR_inter + 'tanh_history_' + dataset + '_LSTM.npy', allow_pickle=True).item()
    print("Max val_accuracy", dataset, ": ", max(LSTM_performance[dataset]["val_accuracy"]))

#%% Plot train and validation performance LSTM

datasets = ['zheng', 'baron_human', 'amb', 'inter_pbmc']

for dataset in datasets:
    plt.plot(LSTM_performance[dataset]["val_accuracy"])

plt.title('LSTM accuracy')
plt.xlim([0.0, 105])
plt.ylim([0.0, 1.05])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(datasets, loc='upper left')
plt.savefig('Figures/History LSTM.png', dpi=300)
plt.show()





#%%###########################
##### Hybrid performance #####
##############################

# Load train and validation performance of hybrid

datasets = ['zheng', 'baron_human', 'amb']

hybrid_performance = {}

for dataset in datasets:
    hybrid_performance[dataset] = np.load(cleandataDIR_intra + 'tanh_history_' + dataset + '.npy', allow_pickle=True).item()
    print("Max val_accuracy", dataset, ": ", max(hybrid_performance[dataset]["val_accuracy"]))

datasets = ['inter_pbmc']

for dataset in datasets:
    hybrid_performance[dataset] = np.load(cleandataDIR_inter + 'tanh_history_' + dataset + '.npy', allow_pickle=True).item()
    print("Max val_accuracy", dataset, ": ", max(hybrid_performance[dataset]["val_accuracy"]))

#%% Plot train and validation performance hybrid

datasets = ['zheng', 'baron_human', 'amb', 'inter_pbmc']

for dataset in datasets:
    plt.plot(hybrid_performance[dataset]["val_accuracy"])

plt.title('Hybrid accuracy')
plt.xlim([0.0, 105])
plt.ylim([0.0, 1.05])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(datasets, loc='upper left')
plt.savefig('Figures/History hybrid.png', dpi=300)
plt.show()