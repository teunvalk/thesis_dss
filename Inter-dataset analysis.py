# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:11:37 2022

@author: teun_
"""

import pandas as pd
import os
import timeit
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.manifold import TSNE

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Embedding, Bidirectional, LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

#%% Set working directory

os.chdir("C:/Users/teun_/OneDrive/Documenten/Master Data Science and Society/Thesis")
datadir_inter = os.getcwd().replace("\\", "/") + "/Data/Inter-dataset/"
cleandataDIR = os.getcwd().replace("\\", "/") + "/Data/Clean/Inter-datasets/"
modelDIR = "C:/Users/teun_/OneDrive/Documenten/Master Data Science and Society/Thesis/Models/"

#%% Read csv and convert to feather data

# Read data
data_pbmc_10Xv2 = pd.read_csv(datadir_inter + "PbmcBench/10Xv2/10Xv2_pbmc1.csv")
data_pbmc_10Xv3 = pd.read_csv(datadir_inter + "PbmcBench/10Xv3/10Xv3_pbmc1.csv")
data_pbmc_CELseq = pd.read_csv(datadir_inter + "PbmcBench/CEL-Seq/CL_pbmc1.csv")
data_pbmc_DropSeq = pd.read_csv(datadir_inter + "PbmcBench/Drop-Seq/DR_pbmc1.csv")

# Read labels
labels_pbmc_10Xv2 = pd.read_csv(datadir_inter + "PbmcBench/10Xv2/10Xv2_pbmc1Labels.csv")
labels_pbmc_10Xv3 = pd.read_csv(datadir_inter + "PbmcBench/10Xv3/10Xv3_pbmc1Labels.csv")
labels_pbmc_CELseq = pd.read_csv(datadir_inter + "PbmcBench/CEL-Seq/CL_pbmc1Labels.csv")
labels_pbmc_DropSeq = pd.read_csv(datadir_inter + "PbmcBench/Drop-Seq/DR_pbmc1Labels.csv")

# Save datasets as feather file
data_pbmc_10Xv2.to_feather(cleandataDIR + "10Xv2.feather")
data_pbmc_10Xv3.to_feather(cleandataDIR + "10Xv3.feather")
data_pbmc_CELseq.to_feather(cleandataDIR + "CELseq.feather")
data_pbmc_DropSeq.to_feather(cleandataDIR + "ropSeq.feather")

# Save labels as feather file
labels_pbmc_10Xv2.to_feather(cleandataDIR + "10Xv2_labels.feather")
labels_pbmc_10Xv3.to_feather(cleandataDIR + "10Xv3_labels.feather")
labels_pbmc_CELseq.to_feather(cleandataDIR + "CELseq_labels.feather")
labels_pbmc_DropSeq.to_feather(cleandataDIR + "DropSeq_labels.feather")

#%% Read feather data

# Read data
data_pbmc_10Xv2 = pd.read_feather(cleandataDIR + "10Xv2.feather")
data_pbmc_10Xv3 = pd.read_feather(cleandataDIR + "10Xv3.feather")
data_pbmc_CELseq = pd.read_feather(cleandataDIR + "CELseq.feather")
data_pbmc_DropSeq = pd.read_feather(cleandataDIR + "DropSeq.feather")

# Read labels
labels_pbmc_10Xv2 = pd.read_feather(cleandataDIR + "10Xv2_labels.feather")
labels_pbmc_10Xv3 = pd.read_feather(cleandataDIR + "10Xv3_labels.feather")
labels_pbmc_CELseq = pd.read_feather(cleandataDIR + "CELseq_labels.feather")
labels_pbmc_DropSeq = pd.read_feather(cleandataDIR + "DropSeq_labels.feather")

#%% EDA

# Get cell count
count_labels_pbmc_10Xv2 = labels_pbmc_10Xv2.x.value_counts().reset_index().rename(columns={"index": "Cell type", "x": "Count"})
count_labels_pbmc_10Xv3 = labels_pbmc_10Xv3.x.value_counts().reset_index().rename(columns={"index": "Cell type", "x": "Count"})
count_labels_pbmc_CELseq = labels_pbmc_CELseq.x.value_counts().reset_index().rename(columns={"index": "Cell type", "x": "Count"})
count_labels_pbmc_DropSeq = labels_pbmc_DropSeq.x.value_counts().reset_index().rename(columns={"index": "Cell type", "x": "Count"})

# Plot cell counts
labels_pbmc_10Xv2["x"].value_counts().plot(kind="barh")
labels_pbmc_10Xv3["x"].value_counts().plot(kind="barh")
labels_pbmc_CELseq["x"].value_counts().plot(kind="barh")
labels_pbmc_DropSeq["x"].value_counts().plot(kind="barh")

# Create tSNE plots
pca = PCA(n_components=1000)

pca_result_1000_10Xv2 = pca.fit_transform(data_pbmc_10Xv2.drop("Unnamed: 0", axis=1).values)
pca_result_1000_10Xv3 = pca.fit_transform(data_pbmc_10Xv3.drop("Unnamed: 0", axis=1).values)
pca_result_1000_CELseq = pca.fit_transform(data_pbmc_CELseq.drop("Unnamed: 0", axis=1).values)
pca_result_1000_DropSeq = pca.fit_transform(data_pbmc_DropSeq.drop("Unnamed: 0", axis=1).values)

np.savez(cleandataDIR + "PCA1000_inter.npz", pca_result_1000_10Xv2, pca_result_1000_10Xv3, pca_result_1000_CELseq, pca_result_1000_DropSeq)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results_10Xv2 = tsne.fit_transform(pca_result_1000_10Xv2)
tsne_results_10Xv3 = tsne.fit_transform(pca_result_1000_10Xv3)
tsne_results_CELseq = tsne.fit_transform(pca_result_1000_CELseq)
tsne_results_DropSeq = tsne.fit_transform(pca_result_1000_DropSeq)

np.savez(cleandataDIR + "tsne_inter.npz", tsne_results_10Xv2, tsne_results_10Xv3, tsne_results_CELseq, tsne_results_DropSeq)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results_10Xv2[:,0], y=tsne_results_10Xv2[:,1],
    hue=labels_pbmc_10Xv2["x"],
    palette=sns.color_palette("hls", 9),
    legend="full",
    alpha=0.3
)
plt.savefig('Figures/tsne inter 10Xv2.png', dpi=300)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results_10Xv3[:,0], y=tsne_results_10Xv3[:,1],
    hue=labels_pbmc_10Xv3["x"],
    palette=sns.color_palette("hls", 8),
    legend="full",
    alpha=0.3
)
plt.savefig('Figures/tsne inter 10Xv3.png', dpi=300)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results_CELseq[:,0], y=tsne_results_CELseq[:,1],
    hue=labels_pbmc_CELseq["x"],
    palette=sns.color_palette("hls", 7),
    legend="full",
    alpha=0.3
)
plt.savefig('Figures/tsne inter CELseq.png', dpi=300)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results_DropSeq[:,0], y=tsne_results_DropSeq[:,1],
    hue=labels_pbmc_DropSeq["x"],
    palette=sns.color_palette("hls", 9),
    legend="full",
    alpha=0.3
)
plt.savefig('Figures/tsne inter DropSeq.png', dpi=300)





#%% Align collumns

# Get column values
cols_pbmc_10Xv2 = data_pbmc_10Xv2.columns
cols_pbmc_10Xv3 = data_pbmc_10Xv3.columns
cols_pbmc_CELseq = data_pbmc_CELseq.columns
cols_pbmc_DropSeq = data_pbmc_DropSeq.columns

# Get intersected columns
cols_intersect = cols_pbmc_10Xv2.intersection(cols_pbmc_10Xv3)
cols_intersect = cols_intersect.intersection(cols_pbmc_CELseq)
cols_intersect = cols_intersect.intersection(cols_pbmc_DropSeq)

# Only keep intersect columns in dataset
data_pbmc_10Xv2 = data_pbmc_10Xv2[cols_intersect]
data_pbmc_10Xv3 = data_pbmc_10Xv3[cols_intersect]
data_pbmc_CELseq = data_pbmc_CELseq[cols_intersect]
data_pbmc_DropSeq = data_pbmc_DropSeq[cols_intersect]

# Save datasets as feather file
data_pbmc_10Xv2.to_feather(cleandataDIR + "10Xv2.feather")
data_pbmc_10Xv3.to_feather(cleandataDIR + "10Xv3.feather")
data_pbmc_CELseq.to_feather(cleandataDIR + "CELseq.feather")
data_pbmc_DropSeq.to_feather(cleandataDIR + "DropSeq.feather")

#%% Read feather data

# Read data
data_pbmc_10Xv2 = pd.read_feather(cleandataDIR + "10Xv2.feather")
data_pbmc_10Xv3 = pd.read_feather(cleandataDIR + "10Xv3.feather")
data_pbmc_CELseq = pd.read_feather(cleandataDIR + "CELseq.feather")
data_pbmc_DropSeq = pd.read_feather(cleandataDIR + "DropSeq.feather")

# Read labels
labels_pbmc_10Xv2 = pd.read_feather(cleandataDIR + "10Xv2_labels.feather")
labels_pbmc_10Xv3 = pd.read_feather(cleandataDIR + "10Xv3_labels.feather")
labels_pbmc_CELseq = pd.read_feather(cleandataDIR + "CELseq_labels.feather")
labels_pbmc_DropSeq = pd.read_feather(cleandataDIR + "DropSeq_labels.feather")

#%% Use one hot encoding for labels

labels_pbmc_train = pd.concat([labels_pbmc_10Xv2, labels_pbmc_10Xv3, labels_pbmc_CELseq])
labels_pbmc_test = labels_pbmc_DropSeq

onehot_inter = LabelBinarizer()

onehot_labels_pbmc_train = onehot_inter.fit_transform(labels_pbmc_train)
onehot_labels_pbmc_test = onehot_inter.transform(labels_pbmc_test)

#%% Split data into train & test data

data_pbmc_train = pd.concat([data_pbmc_10Xv2, data_pbmc_10Xv3, data_pbmc_CELseq])
data_pbmc_test = data_pbmc_DropSeq

data_pbmc_train, data_pbmc_val, onehot_labels_pbmc_train, onehot_labels_pbmc_val = train_test_split(
    data_pbmc_train, onehot_labels_pbmc_train, test_size=0.2, random_state=123, shuffle=True)

labels_pbmc_train, labels_pbmc_val = train_test_split(
    labels_pbmc_train, test_size=0.2, random_state=123, shuffle=True)

#%% Normalize data

def NormalizeData(train, val, test):
    
    train = train.set_index("Unnamed: 0").T
    val = val.set_index("Unnamed: 0").T
    test = test.set_index("Unnamed: 0").T
    
    train_out = np.log2(1 + (train * 1e6) / train.sum())
    val_out = np.log2(1 + (val * 1e6) / val.sum())
    test_out = np.log2(1 + (test * 1e6) / test.sum())

    return train_out.T.reset_index(), val_out.T.reset_index(), test_out.T.reset_index()

data_pbmc_train, data_pbmc_val, data_pbmc_test = NormalizeData(
    data_pbmc_train, data_pbmc_val, data_pbmc_test)

#%% Create numpy arrays

data_pbmc_train = data_pbmc_train.drop("Unnamed: 0", axis=1).to_numpy()
data_pbmc_val = data_pbmc_val.drop("Unnamed: 0", axis=1).to_numpy()
data_pbmc_test = data_pbmc_test.drop("Unnamed: 0", axis=1).to_numpy()

labels_pbmc_train = labels_pbmc_train.to_numpy()
labels_pbmc_val = labels_pbmc_val.to_numpy()
labels_pbmc_test = labels_pbmc_test.to_numpy()

#%% Save as npz files

np.savez(cleandataDIR + "pbmc_train_data_label.npz", data_pbmc_train, labels_pbmc_train)
np.savez(cleandataDIR + "pbmc_val_data_label.npz", data_pbmc_val, labels_pbmc_val)
np.savez(cleandataDIR + "pbmc_test_data_label.npz", data_pbmc_test, labels_pbmc_test)

#%% Read npz files

with np.load(cleandataDIR + "pbmc_train_data_label.npz", allow_pickle=True) as data:
    data_pbmc_train = data["arr_0"]
    labels_pbmc_train = data["arr_1"]

with np.load(cleandataDIR + "pbmc_val_data_label.npz", allow_pickle=True) as data:
    data_pbmc_val = data["arr_0"]
    labels_pbmc_val = data["arr_1"]

with np.load(cleandataDIR + "pbmc_test_data_label.npz", allow_pickle=True) as data:
    data_pbmc_test = data["arr_0"]
    labels_pbmc_test = data["arr_1"]
    
with np.load(cleandataDIR + "pbmc_labels_onehot.npz", allow_pickle=True) as data:
    onehot_labels_pbmc_train = data["arr_0"]
    onehot_labels_pbmc_val = data["arr_1"]
    onehot_labels_pbmc_test = data["arr_2"]





#%%############
##### SVM #####
###############

start = timeit.default_timer()

svm_model = svm.SVC()

svm_model.fit(data_pbmc_train, labels_pbmc_train)

pickle.dump(svm_model, open(modelDIR + 'inter_svm.sav', 'wb'))

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))

#%% Check performance SVM

start = timeit.default_timer()

svm_model = pickle.load(open(modelDIR + 'inter_svm.sav', 'rb'))

y_true_val, y_pred_val = labels_pbmc_val, svm_model.predict(data_pbmc_val)
acc_val = accuracy_score(y_true_val, y_pred_val)

y_true_test, y_pred_test = labels_pbmc_test, svm_model.predict(data_pbmc_test)
acc_test = accuracy_score(y_true_test, y_pred_test)

np.savez(cleandataDIR + "inter_svm_pred_val.npz", y_true_val, y_pred_val)
np.savez(cleandataDIR + "inter_svm_pred_test.npz", y_true_test, y_pred_test)

with np.load(cleandataDIR + "inter_svm_pred_val.npz", allow_pickle=True) as data:
    y_true_val = data["arr_0"]
    y_pred_val = data["arr_1"]

with np.load(cleandataDIR + "inter_svm_pred_test.npz", allow_pickle=True) as data:
    y_true_test = data["arr_0"]
    y_pred_test = data["arr_1"]

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))





#%%###################
##### Linear SVM #####
######################

start = timeit.default_timer()

linearsvm_model = svm.LinearSVC()

linearsvm_model.fit(data_pbmc_train, labels_pbmc_train)

pickle.dump(linearsvm_model, open(modelDIR + 'inter_linear_svm.sav', 'wb'))

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))

#%% SVM performance

start = timeit.default_timer()

linearsvm_model = pickle.load(open(modelDIR + 'inter_linear_svm.sav', 'rb'))

y_true_val, y_pred_val = labels_pbmc_val, linearsvm_model.predict(data_pbmc_val)
acc_val = accuracy_score(y_true_val, y_pred_val)

y_true_test, y_pred_test = labels_pbmc_test, linearsvm_model.predict(data_pbmc_test)
acc_test = accuracy_score(y_true_test, y_pred_test)

np.savez(cleandataDIR + "inter_linear_svm_pred_val.npz", y_true_val, y_pred_val)
np.savez(cleandataDIR + "inter_linear_svm_pred_test.npz", y_true_test, y_pred_test)

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))





#%%############
##### CNN #####
###############

def create_model_CNN(activation='tanh', dropout_rate=0.5, optimizer='Adam'):
    model = Sequential()
    
    model.add(Conv1D(64, kernel_size=3, activation=activation, input_shape=(17751,1)))
    
    model.add(MaxPooling1D(9))
    
    model.add(Conv1D(64, kernel_size=3, activation=activation))
    
    model.add(MaxPooling1D(9))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(64, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(32, activation=activation))
    
    model.add(Dense(9, activation='softmax'))
    
    model.compile(optimizer='Adamax',  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model_CNN()

model.summary()

#%% Tune activation function

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

activation = ['softsign', 'relu', 'tanh', 'linear']

activation = ['linear']

for act_function in activation:
    print("Now running:", act_function)
    
    tf.random.set_seed(123)
    
    model = create_model_CNN(activation=act_function)
    
    history = model.fit(data_pbmc_train, onehot_labels_pbmc_train,
                        validation_data=(data_pbmc_val, onehot_labels_pbmc_val),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "inter_pbmc_cnn_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_inter_pbmc_cnn.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

activation = ['softplus', 'softsign', 'relu', 'tanh', 'linear']

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_inter_pbmc_cnn.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))







#%%#######################
##### Gene sentences #####
##########################

# Read data
data_pbmc_10Xv2 = pd.read_feather(cleandataDIR + "10Xv2.feather")
data_pbmc_10Xv3 = pd.read_feather(cleandataDIR + "10Xv3.feather")
data_pbmc_CELseq = pd.read_feather(cleandataDIR + "CELseq.feather")
data_pbmc_DropSeq = pd.read_feather(cleandataDIR + "DropSeq.feather")

data_pbmc_10Xv2 = data_pbmc_10Xv2.drop("Unnamed: 0", axis=1)
data_pbmc_10Xv3 = data_pbmc_10Xv3.drop("Unnamed: 0", axis=1)
data_pbmc_CELseq = data_pbmc_CELseq.drop("Unnamed: 0", axis=1)
data_pbmc_DropSeq = data_pbmc_DropSeq.drop("Unnamed: 0", axis=1)

data_pbmc_train = pd.concat([data_pbmc_10Xv2, data_pbmc_10Xv3, data_pbmc_CELseq])
data_pbmc_test = data_pbmc_DropSeq

# Define function to create sentences
def CreateGeneSentence(df):
    genes = df.columns
    sentences = []
    
    for x in df.values >= 1:
        sentences.append(" ".join(list(genes[x])))
    
    return np.asarray(sentences)

sentences_pbmc_train = CreateGeneSentence(data_pbmc_train)
sentences_pbmc_test = CreateGeneSentence(data_pbmc_test)

#%% Split into train, validation and test data

sentences_pbmc_train, sentences_pbmc_val = train_test_split(
    sentences_pbmc_train, test_size=0.2, random_state=123, shuffle=True)

#%% Save to npz files

np.savez(cleandataDIR + "sentences_train_val_test.npz", sentences_pbmc_train, sentences_pbmc_val, sentences_pbmc_test)

#%% Read npz file

with np.load(cleandataDIR + "sentences_train_val_test.npz", allow_pickle=True) as data:
    sentences_pbmc_train = data["arr_0"]
    sentences_pbmc_val = data["arr_1"]
    sentences_pbmc_test = data["arr_2"]

with np.load(cleandataDIR + "pbmc_labels_onehot.npz", allow_pickle=True) as data:
    onehot_labels_pbmc_train = data["arr_0"]
    onehot_labels_pbmc_val = data["arr_1"]
    onehot_labels_pbmc_test = data["arr_2"]





#%%#############
##### LSTM #####
################

# Use Word2Vec

VOCAB_SIZE = 17751

encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, standardize=None)
encoder.adapt(sentences_pbmc_train)

#%% Train model

def create_model_LSTM(activation='tanh', dropout_rate=0.5, optimizer='Adam'):
    model = Sequential()
    
    model.add(encoder)
    
    model.add(Embedding(input_dim=len(encoder.get_vocabulary()),
        output_dim=256,
        mask_zero=True))
    
    model.add(Bidirectional(LSTM(256)))
    
    model.add(Dense(256, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(64, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(32, activation=activation))
    
    model.add(Dense(9, activation='softmax'))
    
    model.compile(optimizer=optimizer,  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model_LSTM()

model.summary()

#%% Tune activation function

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

activation = ['tanh']

for act_function in activation:
    print("Now running:", act_function)
    
    tf.random.set_seed(123)
    
    model = create_model_LSTM(activation=act_function)
    
    history = model.fit(sentences_pbmc_train, onehot_labels_pbmc_train,
                        validation_data=(sentences_pbmc_val, onehot_labels_pbmc_val),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "inter_pbmc_lstm_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_inter_pbmc_lstm.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_inter_pbmc_lstm.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))





#%%##################################
##### CNN + LSTM with embedding #####
#####################################

def create_model_CNN_LSTM(activation='tanh', dropout_rate=0.5, optimizer='Adam'):
    model = Sequential()
    
    model.add(encoder)
    
    model.add(Embedding(input_dim=len(encoder.get_vocabulary()),
        output_dim=256,
        mask_zero=True))
    
    model.add(Conv1D(64, kernel_size=3, activation=activation))
    
    model.add(Conv1D(64, kernel_size=3, activation=activation))
    
    model.add(Bidirectional(LSTM(256)))
    
    model.add(Dense(256, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(64, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(32, activation=activation))
    
    model.add(Dense(9, activation='softmax'))
    
    model.compile(optimizer=optimizer,  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model_CNN_LSTM()

model.summary()

#%% Tune activation function

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

activation = ['tanh']

for act_function in activation:
    print("Now running:", act_function)
    
    tf.random.set_seed(123)
    
    model = create_model_CNN_LSTM(activation=act_function)
    
    history = model.fit(sentences_pbmc_train, onehot_labels_pbmc_train,
                        validation_data=(sentences_pbmc_val, onehot_labels_pbmc_val),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "inter_pbmc_cnn_lstm_embedding_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_inter_pbmc.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_inter_pbmc.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))

