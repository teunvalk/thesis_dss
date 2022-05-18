# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:14:59 2022

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
datadir_intra = os.getcwd().replace("\\", "/") + "/Data/Intra-dataset/"
cleandataDIR = os.getcwd().replace("\\", "/") + "/Data/Clean/Intra-datasets/"
modelDIR = "C:/Users/teun_/OneDrive/Documenten/Master Data Science and Society/Thesis/Models/"

#%% Read csv and convert to feather data

# Read data
data_zheng = pd.read_csv(datadir_intra + "Zheng 68K/Filtered_68K_PBMC_data.csv")

# Read labels
labels_zheng = pd.read_csv(datadir_intra + "Zheng 68K/Labels.csv")

# Save datasets as feather file
data_zheng.to_feather(cleandataDIR + "zheng_68K.feather")

# Save labels as feather file
labels_zheng.to_feather(cleandataDIR + "zheng_68K_labels.feather")

#%% Read feather data

# Read data
data_zheng = pd.read_feather(cleandataDIR + "zheng_68K.feather")

# Read labels
labels_zheng = pd.read_feather(cleandataDIR + "zheng_68K_labels.feather")

#%% EDA

# Get cell count
count_labels_zheng = labels_zheng.x.value_counts().reset_index().rename(columns={"index": "Cell type", "x": "Count"})

# Plot cell counts
labels_zheng["x"].value_counts().plot(kind="barh")

# Create tSNE plot
pca = PCA(n_components=1000)
pca_result = pca.fit_transform(data_zheng.drop("Unnamed: 0", axis=1).values)

np.savez(cleandataDIR + "PCA1000_zheng.npz", pca_result)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result)

np.savez(cleandataDIR + "tsne_zheng.npz", tsne_results)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels_zheng["x"],
    palette=sns.color_palette("hls", 11),
    legend="full",
    alpha=0.3
)
plt.savefig('Figures/tsne zheng.png', dpi=300)





#%% Use one hot encoding for labels

onehot_zheng = LabelBinarizer()
onehot_labels_zheng = onehot_zheng.fit_transform(labels_zheng)

#%% Split data into train & test data

data_train_zheng, data_test_zheng, onehot_labels_train_zheng, onehot_labels_test_zheng = train_test_split(
    data_zheng, onehot_labels_zheng, test_size=0.2, random_state=123, shuffle=True)

data_train_zheng, data_val_zheng, onehot_labels_train_zheng, onehot_labels_val_zheng = train_test_split(
    data_train_zheng, onehot_labels_train_zheng, test_size=0.25, random_state=123, shuffle=True)

labels_train_zheng, labels_test_zheng = train_test_split(
    labels_zheng, test_size=0.2, random_state=123, shuffle=True)

labels_train_zheng, labels_val_zheng = train_test_split(
    labels_train_zheng, test_size=0.25, random_state=123, shuffle=True)

#%% Normalize data

def NormalizeData(train, val, test):
    
    train = train.set_index("Unnamed: 0").T
    val = val.set_index("Unnamed: 0").T
    test = test.set_index("Unnamed: 0").T
    
    train_out = np.log2(1 + (train * 1e6) / train.sum())
    val_out = np.log2(1 + (val * 1e6) / val.sum())
    test_out = np.log2(1 + (test * 1e6) / test.sum())
    
    return train_out.T.reset_index(), val_out.T.reset_index(), test_out.T.reset_index()

data_train_zheng, data_val_zheng, data_test_zheng = NormalizeData(
    data_train_zheng, data_val_zheng, data_test_zheng)

#%% Create numpy arrays

data_train_zheng = data_train_zheng.drop("Unnamed: 0", axis=1).to_numpy()
data_val_zheng = data_val_zheng.drop("Unnamed: 0", axis=1).to_numpy()
data_test_zheng = data_test_zheng.drop("Unnamed: 0", axis=1).to_numpy()

labels_train_zheng = labels_train_zheng.to_numpy()
labels_val_zheng = labels_val_zheng.to_numpy()
labels_test_zheng = labels_test_zheng.to_numpy()

#%% Save as npz files

np.savez(cleandataDIR + "zheng_train_data_label.npz", data_train_zheng, labels_train_zheng)
np.savez(cleandataDIR + "zheng_val_data_label.npz", data_val_zheng, labels_val_zheng)
np.savez(cleandataDIR + "zheng_test_data_label.npz", data_test_zheng, labels_test_zheng)

#%% Read npz files

with np.load(cleandataDIR + "zheng_train_data_label.npz", allow_pickle=True) as data:
    data_train_zheng = data["arr_0"]
    labels_train_zheng = data["arr_1"]

with np.load(cleandataDIR + "zheng_val_data_label.npz", allow_pickle=True) as data:
    data_val_zheng = data["arr_0"]
    labels_val_zheng = data["arr_1"]

with np.load(cleandataDIR + "zheng_test_data_label.npz", allow_pickle=True) as data:
    data_test_zheng = data["arr_0"]
    labels_test_zheng = data["arr_1"]
    
with np.load(cleandataDIR + "zheng_labels_onehot.npz", allow_pickle=True) as data:
    onehot_labels_train_zheng = data["arr_0"]
    onehot_labels_val_zheng = data["arr_1"]
    onehot_labels_test_zheng = data["arr_2"]





#%%############
##### SVM #####
###############

start = timeit.default_timer()

svm_model = svm.SVC()

svm_model.fit(data_train_zheng, labels_train_zheng)

pickle.dump(svm_model, open(modelDIR + 'intra_svm_zheng.sav', 'wb'))

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))

#%% Check performance SVM

start = timeit.default_timer()

svm_model = pickle.load(open(modelDIR + 'intra_svm_zheng.sav', 'rb'))

y_true_val, y_pred_val = labels_val_zheng, svm_model.predict(data_val_zheng)
acc_val = accuracy_score(y_true_val, y_pred_val)

y_true_test, y_pred_test = labels_test_zheng, svm_model.predict(data_test_zheng)
acc_test = accuracy_score(y_true_test, y_pred_test)

np.savez(cleandataDIR + "intra_svm_pred_val_zheng.npz", y_true_val, y_pred_val)
np.savez(cleandataDIR + "intra_svm_pred_test_zheng.npz", y_true_test, y_pred_test)

with np.load(cleandataDIR + "intra_svm_pred_val_zheng.npz", allow_pickle=True) as data:
    y_true_val = data["arr_0"]
    y_pred_val = data["arr_1"]

with np.load(cleandataDIR + "intra_svm_pred_test_zheng.npz", allow_pickle=True) as data:
    y_true_test = data["arr_0"]
    y_pred_test = data["arr_1"]

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))





#%%###################
##### Linear SVM #####
######################

start = timeit.default_timer()

linearsvm_model = svm.LinearSVC()

linearsvm_model.fit(data_train_zheng, labels_train_zheng)

pickle.dump(linearsvm_model, open(modelDIR + 'intra_linear_svm_zheng.sav', 'wb'))

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))

#%% SVM performance

start = timeit.default_timer()

linearsvm_model = pickle.load(open(modelDIR + 'intra_linear_svm_zheng.sav', 'rb'))

y_true_val, y_pred_val = labels_val_zheng, linearsvm_model.predict(data_val_zheng)
acc_val = accuracy_score(y_true_val, y_pred_val)

y_true_test, y_pred_test = labels_test_zheng, linearsvm_model.predict(data_test_zheng)
acc_test = accuracy_score(y_true_test, y_pred_test)

np.savez(cleandataDIR + "intra_linear_svm_pred_val_zheng.npz", y_true_val, y_pred_val)
np.savez(cleandataDIR + "intra_linear_svm_pred_test_zheng.npz", y_true_test, y_pred_test)

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))





#%%############
##### CNN #####
###############

def create_model_CNN(activation='tanh', dropout_rate=0.5, optimizer='Adam'):
    model = Sequential()
    
    model.add(Conv1D(64, kernel_size=3, activation=activation, input_shape=(20387,1)))
    
    model.add(MaxPooling1D(9))
    
    model.add(Conv1D(64, kernel_size=3, activation=activation))
    
    model.add(MaxPooling1D(9))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(64, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(32, activation=activation))
    
    model.add(Dense(11, activation='softmax'))
    
    model.compile(optimizer='Adamax',  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model_CNN()

model.summary()

#%% Tune activation function

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

for act_function in activation:
    print("Now running:", act_function)
    
    tf.random.set_seed(123)
    
    model = create_model_CNN(activation=act_function)
    
    history = model.fit(data_train_zheng, onehot_labels_train_zheng,
                        validation_data=(data_val_zheng, onehot_labels_val_zheng),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "intra_zheng_cnn_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_zheng_cnn.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_zheng_cnn.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))

#%% Check optimal amount of epochs

optimal_activation = 'relu'

print("Max val_accuracy", optimal_activation, ": ", max(act_performance[optimal_activation]["val_accuracy"]))
print("Optimal epoch amount", optimal_activation, ": ", act_performance[optimal_activation]["val_accuracy"].index(max(act_performance[optimal_activation]["val_accuracy"])) + 1)





#%%#######################
##### Gene sentences #####
##########################

# Read data
data_zheng = pd.read_feather(cleandataDIR + "zheng.feather")
data_zheng = data_zheng.drop("Unnamed: 0", axis=1)

# Define function to create sentences
def CreateGeneSentence(df):
    genes = df.columns
    sentences = []
    
    for x in df.values >= 1:
        sentences.append(" ".join(list(genes[x])))
    
    return np.asarray(sentences)

sentences_zheng = CreateGeneSentence(data_zheng)

#%% Split into train, validation and test data

sentences_train_zheng, sentences_test_zheng = train_test_split(
    sentences_zheng, test_size=0.2, random_state=123, shuffle=True)

sentences_train_zheng, sentences_val_zheng = train_test_split(
    sentences_train_zheng, test_size=0.25, random_state=123, shuffle=True)

#%% Save to npz files

np.savez(cleandataDIR + "sentences_zheng_train_val_test.npz", sentences_train_zheng, sentences_val_zheng, sentences_test_zheng)

#%% Read npz file

with np.load(cleandataDIR + "sentences_zheng_train_val_test.npz", allow_pickle=True) as data:
    sentences_train_zheng = data["arr_0"]
    sentences_val_zheng = data["arr_1"]
    sentences_test_zheng = data["arr_2"]

with np.load(cleandataDIR + "zheng_labels_onehot.npz", allow_pickle=True) as data:
    onehot_labels_train_zheng = data["arr_0"]
    onehot_labels_val_zheng = data["arr_1"]
    onehot_labels_test_zheng = data["arr_2"]





#%%#############
##### LSTM #####
################

# Use Word2Vec

VOCAB_SIZE = 20387

encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, standardize=None)
encoder.adapt(sentences_train_zheng)

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
    
    model.add(Dense(11, activation='softmax'))
    
    model.compile(optimizer=optimizer,  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model_LSTM()

model.summary()

#%% Tune activation function

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

for act_function in activation:
    print("Now running:", act_function)
    
    tf.random.set_seed(123)
    
    model = create_model_LSTM(activation=act_function)
    
    history = model.fit(sentences_train_zheng, onehot_labels_train_zheng,
                        validation_data=(sentences_val_zheng, onehot_labels_val_zheng),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "intra_zheng_lstm_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_zheng_lstm.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_zheng_lstm.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))

#%% Check optimal amount of epochs

optimal_activation = 'tanh'

print("Max val_accuracy", optimal_activation, ": ", max(act_performance[optimal_activation]["val_accuracy"]))
print("Optimal epoch amount", optimal_activation, ": ", act_performance[optimal_activation]["val_accuracy"].index(max(act_performance[optimal_activation]["val_accuracy"])) + 1)





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
    
    model.add(Dense(11, activation='softmax'))
    
    model.compile(optimizer=optimizer,  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model_CNN_LSTM()

model.summary()

#%% Tune activation function

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

for act_function in activation:
    print("Now running:", act_function)
    
    tf.random.set_seed(123)
    
    model = create_model_CNN_LSTM(activation=act_function)
    
    history = model.fit(sentences_train_zheng, onehot_labels_train_zheng,
                        validation_data=(sentences_val_zheng, onehot_labels_val_zheng),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "intra_zheng_cnn_lstm_embedding_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_zheng.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_zheng.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))

#%% Check optimal amount of epochs

optimal_activation = 'tanh'

print("Max val_accuracy", optimal_activation, ": ", max(act_performance[optimal_activation]["val_accuracy"]))
print("Optimal epoch amount", optimal_activation, ": ", act_performance[optimal_activation]["val_accuracy"].index(max(act_performance[optimal_activation]["val_accuracy"])) + 1)