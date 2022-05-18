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
data_baron_human = pd.read_csv(datadir_intra + "Pancreatic_data/Baron Human/Filtered_Baron_HumanPancreas_data.csv")

# Read labels
labels_baron_human = pd.read_csv(datadir_intra + "Pancreatic_data/Baron Human/Labels.csv")

# Save datasets as feather file
data_baron_human.to_feather(cleandataDIR + "baron_human.feather")

# Save labels as feather file
labels_baron_human.to_feather(cleandataDIR + "baron_human_labels.feather")

#%% Read feather data

# Read data
data_baron_human = pd.read_feather(cleandataDIR + "baron_human.feather")

# Read labels
labels_baron_human = pd.read_feather(cleandataDIR + "baron_human_labels.feather")

#%% EDA

# Get cell count
count_labels_baron_human = labels_baron_human.x.value_counts().reset_index().rename(columns={"index": "Cell type", "x": "Count"})

# Plot cell counts
labels_baron_human["x"].value_counts().plot(kind="barh")

# Create tSNE or UMAP plots
pca = PCA(n_components=1000)
pca_result = pca.fit_transform(data_baron_human.drop("Unnamed: 0", axis=1).values)

np.savez(cleandataDIR + "PCA1000_baron_human.npz", pca_result)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result)

np.savez(cleandataDIR + "tsne_baron_human.npz", tsne_results)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels_baron_human["x"],
    palette=sns.color_palette("hls", 14),
    legend="full",
    alpha=0.3
)
plt.savefig('Figures/tsne baron_human.png', dpi=300)





#%% Use one hot encoding for labels

onehot_baron_human = LabelBinarizer()
onehot_labels_baron_human = onehot_baron_human.fit_transform(labels_baron_human)

#%% Split data into train & test data

data_train_baron_human, data_test_baron_human, onehot_labels_train_baron_human, onehot_labels_test_baron_human = train_test_split(
    data_baron_human, onehot_labels_baron_human, test_size=0.2, random_state=123, shuffle=True)

data_train_baron_human, data_val_baron_human, onehot_labels_train_baron_human, onehot_labels_val_baron_human = train_test_split(
    data_train_baron_human, onehot_labels_train_baron_human, test_size=0.25, random_state=123, shuffle=True)

labels_train_baron_human, labels_test_baron_human = train_test_split(
    labels_baron_human, test_size=0.2, random_state=123, shuffle=True)

labels_train_baron_human, labels_val_baron_human = train_test_split(
    labels_train_baron_human, test_size=0.25, random_state=123, shuffle=True)

#%% Normalize data

def NormalizeData(train, val, test):
    
    train = train.set_index("Unnamed: 0").T
    val = val.set_index("Unnamed: 0").T
    test = test.set_index("Unnamed: 0").T
    
    train_out = np.log2(1 + (train * 1e6) / train.sum())
    val_out = np.log2(1 + (val * 1e6) / val.sum())
    test_out = np.log2(1 + (test * 1e6) / test.sum())
    
    return train_out.T.reset_index(), val_out.T.reset_index(), test_out.T.reset_index()

data_train_baron_human, data_val_baron_human, data_test_baron_human = NormalizeData(
    data_train_baron_human, data_val_baron_human, data_test_baron_human)

#%% Create numpy arrays

data_train_baron_human = data_train_baron_human.drop("Unnamed: 0", axis=1).to_numpy()
data_val_baron_human = data_val_baron_human.drop("Unnamed: 0", axis=1).to_numpy()
data_test_baron_human = data_test_baron_human.drop("Unnamed: 0", axis=1).to_numpy()

labels_train_baron_human = labels_train_baron_human.to_numpy()
labels_val_baron_human = labels_val_baron_human.to_numpy()
labels_test_baron_human = labels_test_baron_human.to_numpy()

#%% Save as npz files

np.savez(cleandataDIR + "baron_human_train_data_label.npz", data_train_baron_human, labels_train_baron_human)
np.savez(cleandataDIR + "baron_human_val_data_label.npz", data_val_baron_human, labels_val_baron_human)
np.savez(cleandataDIR + "baron_human_test_data_label.npz", data_test_baron_human, labels_test_baron_human)

#%% Read npz files

with np.load(cleandataDIR + "baron_human_train_data_label.npz", allow_pickle=True) as data:
    data_train_baron_human = data["arr_0"]
    labels_train_baron_human = data["arr_1"]

with np.load(cleandataDIR + "baron_human_val_data_label.npz", allow_pickle=True) as data:
    data_val_baron_human = data["arr_0"]
    labels_val_baron_human = data["arr_1"]

with np.load(cleandataDIR + "baron_human_test_data_label.npz", allow_pickle=True) as data:
    data_test_baron_human = data["arr_0"]
    labels_test_baron_human = data["arr_1"]
    
with np.load(cleandataDIR + "baron_human_labels_onehot.npz", allow_pickle=True) as data:
    onehot_labels_train_baron_human = data["arr_0"]
    onehot_labels_val_baron_human = data["arr_1"]
    onehot_labels_test_baron_human = data["arr_2"]





#%%############
##### SVM #####
###############

start = timeit.default_timer()

svm_model = svm.SVC(probability=True)

svm_model.fit(data_train_baron_human, labels_train_baron_human)

pickle.dump(svm_model, open(modelDIR + 'intra_svm_baron_human.sav', 'wb'))

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))

#%% Check performance SVM

start = timeit.default_timer()

svm_model = pickle.load(open(modelDIR + 'intra_svm_baron_human.sav', 'rb'))

y_true_val, y_pred_val = labels_val_baron_human, svm_model.predict(data_val_baron_human)
acc_val = accuracy_score(y_true_val, y_pred_val)

y_true_test, y_pred_test = labels_test_baron_human, svm_model.predict(data_test_baron_human)
acc_test = accuracy_score(y_true_test, y_pred_test)

np.savez(cleandataDIR + "intra_svm_pred_val_baron_human.npz", y_true_val, y_pred_val)
np.savez(cleandataDIR + "intra_svm_pred_test_baron_human.npz", y_true_test, y_pred_test)

with np.load(cleandataDIR + "intra_svm_pred_val_baron_human.npz", allow_pickle=True) as data:
    y_true_val = data["arr_0"]
    y_pred_val = data["arr_1"]

with np.load(cleandataDIR + "intra_svm_pred_test_baron_human.npz", allow_pickle=True) as data:
    y_true_test = data["arr_0"]
    y_pred_test = data["arr_1"]

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))





#%%###################
##### Linear SVM #####
######################

start = timeit.default_timer()

linearsvm_model = svm.LinearSVC()

linearsvm_model.fit(data_train_baron_human, labels_train_baron_human)

pickle.dump(linearsvm_model, open(modelDIR + 'intra_linear_svm_baron_human.sav', 'wb'))

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))

#%% SVM performance

start = timeit.default_timer()

linearsvm_model = pickle.load(open(modelDIR + 'intra_linear_svm_baron_human.sav', 'rb'))

y_true_val, y_pred_val = labels_val_baron_human, linearsvm_model.predict(data_val_baron_human)
acc_val = accuracy_score(y_true_val, y_pred_val)

y_true_test, y_pred_test = labels_test_baron_human, linearsvm_model.predict(data_test_baron_human)
acc_test = accuracy_score(y_true_test, y_pred_test)

np.savez(cleandataDIR + "intra_linear_svm_pred_val_baron_human.npz", y_true_val, y_pred_val)
np.savez(cleandataDIR + "intra_linear_svm_pred_test_baron_human.npz", y_true_test, y_pred_test)

stop = timeit.default_timer()
print('Time: {:.2f} seconds'.format(stop - start))





#%%############
##### CNN #####
###############

def create_model_CNN(activation='tanh', dropout_rate=0.5, optimizer='Adam'):
    model = Sequential()
    
    model.add(Conv1D(64, kernel_size=3, activation=activation, input_shape=(17499,1)))
    
    model.add(MaxPooling1D(9))
    
    model.add(Conv1D(64, kernel_size=3, activation=activation))
    
    model.add(MaxPooling1D(9))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(64, activation=activation))
    
    model.add(Dropout(dropout_rate, seed=123))
    
    model.add(Dense(32, activation=activation))
    
    model.add(Dense(14, activation='softmax'))
    
    model.compile(optimizer=optimizer,  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model_CNN()

model.summary()

#%% Tune activation function

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

activation = ['relu', 'linear']

for act_function in activation:
    print("Now running:", act_function)
    
    tf.random.set_seed(123)
    
    model = create_model_CNN(activation=act_function)
    
    history = model.fit(data_train_baron_human, onehot_labels_train_baron_human,
                        validation_data=(data_val_baron_human, onehot_labels_val_baron_human),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "intra_baron_human_cnn_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_baron_human_cnn.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

activation = ['softplus', 'softsign', 'relu', 'tanh', 'linear']

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_baron_human_cnn.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))







#%%#######################
##### Gene sentences #####
##########################

# Read data
data_baron_human = pd.read_feather(cleandataDIR + "baron_human.feather")
data_baron_human = data_baron_human.drop("Unnamed: 0", axis=1)

# Define function to create sentences
def CreateGeneSentence(df):
    genes = df.columns
    sentences = []
    
    for x in df.values >= 1:
        sentences.append(" ".join(list(genes[x])))
    
    return np.asarray(sentences)

sentences_baron_human = CreateGeneSentence(data_baron_human)

#%% Split into train, validation and test data

sentences_train_baron_human, sentences_test_baron_human = train_test_split(
    sentences_baron_human, test_size=0.2, random_state=123, shuffle=True)

sentences_train_baron_human, sentences_val_baron_human = train_test_split(
    sentences_train_baron_human, test_size=0.25, random_state=123, shuffle=True)

#%% Save to npz files

np.savez(cleandataDIR + "sentences_baron_human_train_val_test.npz", sentences_train_baron_human, sentences_val_baron_human, sentences_test_baron_human)

#%% Read npz file

with np.load(cleandataDIR + "sentences_baron_human_train_val_test.npz", allow_pickle=True) as data:
    sentences_train_baron_human = data["arr_0"]
    sentences_val_baron_human = data["arr_1"]
    sentences_test_baron_human = data["arr_2"]

with np.load(cleandataDIR + "baron_human_labels_onehot.npz", allow_pickle=True) as data:
    onehot_labels_train_baron_human = data["arr_0"]
    onehot_labels_val_baron_human = data["arr_1"]
    onehot_labels_test_baron_human = data["arr_2"]





#%%#############
##### LSTM #####
################

# Use Word2Vec

VOCAB_SIZE = 17499

encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, standardize=None)
encoder.adapt(sentences_train_baron_human)

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
    
    model.add(Dense(14, activation='softmax'))
    
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
    
    history = model.fit(sentences_train_baron_human, onehot_labels_train_baron_human,
                        validation_data=(sentences_val_baron_human, onehot_labels_val_baron_human),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "intra_baron_human_lstm_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_baron_human_lstm.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_baron_human_lstm.npy', allow_pickle=True).item()
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
    
    model.add(Dense(14, activation='softmax'))
    
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
    
    history = model.fit(sentences_train_baron_human, onehot_labels_train_baron_human,
                        validation_data=(sentences_val_baron_human, onehot_labels_val_baron_human),
                        epochs=100, batch_size=16)
    
    model.save(modelDIR + "intra_baron_human_cnn_lstm_embedding_" + act_function)
    
    np.save(cleandataDIR + act_function + '_history_baron_human.npy',history.history)
    
    tf.keras.backend.clear_session()

#%% Check performance of activation functions

act_performance = {}

for act_function in activation:
    act_performance[act_function] = np.load(cleandataDIR + act_function + '_history_baron_human.npy', allow_pickle=True).item()
    print("Max val_accuracy", act_function, ": ", max(act_performance[act_function]["val_accuracy"]))

