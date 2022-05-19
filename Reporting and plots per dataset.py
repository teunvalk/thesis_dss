# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:19:58 2022

@author: teun_
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, auc
from scipy import interp
from itertools import cycle

#%% Set working directory

os.chdir("C:/Users/teun_/OneDrive/Documenten/Master Data Science and Society/Thesis")
datadir_inter = os.getcwd().replace("\\", "/") + "/Data/Inter-dataset/"
cleandataDIR_inter = os.getcwd().replace("\\", "/") + "/Data/Clean/Inter-datasets/"
cleandataDIR_intra = os.getcwd().replace("\\", "/") + "/Data/Clean/Intra-datasets/"
modelDIR = "C:/Users/teun_/OneDrive/Documenten/Master Data Science and Society/Thesis/Models/"


def get_performance_and_model(dataset, datadir=cleandataDIR_intra, analysis='intra'):
    dict_out = {}
    
    dict_out["CNN"] = np.load(datadir + 'relu_history_' + dataset + '_cnn.npy', allow_pickle=True).item()
    dict_out["LSTM"] = np.load(datadir + 'tanh_history_' + dataset + '_lstm.npy', allow_pickle=True).item()
    dict_out["Hybrid"] = np.load(datadir + 'tanh_history_' + dataset + '.npy', allow_pickle=True).item()
    
    if analysis == 'intra':
        cnn_model = tf.keras.models.load_model(modelDIR + analysis + '_' + dataset + "_cnn_relu")
        lstm_model = tf.keras.models.load_model(modelDIR + analysis + '_' + dataset + "_lstm_tanh")
        hybrid_model = tf.keras.models.load_model(modelDIR + analysis + '_' + dataset + "_cnn_lstm_embedding_tanh")
        svm_model = pickle.load(open(modelDIR + 'intra_svm_' + dataset + '.sav', 'rb'))
        linearsvm_model = pickle.load(open(modelDIR + 'intra_linear_svm_' + dataset + '.sav', 'rb'))
    
    if analysis == 'inter':
        cnn_model = tf.keras.models.load_model(modelDIR + dataset + "_cnn_relu")
        lstm_model = tf.keras.models.load_model(modelDIR + dataset + "_lstm_tanh")
        hybrid_model = tf.keras.models.load_model(modelDIR + dataset + "_cnn_lstm_embedding_tanh")
        svm_model = pickle.load(open(modelDIR + 'inter_svm.sav', 'rb'))
        linearsvm_model = pickle.load(open(modelDIR + 'inter_linear_svm.sav', 'rb'))
    
    return dict_out, cnn_model, lstm_model, hybrid_model, svm_model, linearsvm_model

def get_data(dataset, datadir=cleandataDIR_intra):
    with np.load(datadir + dataset + "_test_data_label.npz", allow_pickle=True) as data:
        data_test = data["arr_0"]
        labels_test = data["arr_1"]
        
    with np.load(datadir + dataset + "_labels_onehot.npz", allow_pickle=True) as data:
        onehot_labels_test = data["arr_2"]
    
    if datadir == cleandataDIR_intra:
        with np.load(datadir + "sentences_" + dataset + "_train_val_test.npz", allow_pickle=True) as data:
            sentences_test = data["arr_2"]
    
    if datadir == cleandataDIR_inter:
        with np.load(datadir + "sentences_train_val_test.npz", allow_pickle=True) as data:
            sentences_test = data["arr_2"]
    
    return data_test, sentences_test, labels_test, onehot_labels_test





#%%##############
##### Zheng #####
#################

zheng_performance, zheng_model_cnn, zheng_model_lstm, zheng_model_hybrid, zheng_model_svm, zheng_model_linearsvm = get_performance_and_model(
    "zheng", cleandataDIR_intra, 'intra')

zheng_data, zheng_sentences, zheng_labels, zheng_labels_onehot = get_data(
    "zheng", cleandataDIR_intra)

#%% SVM zheng

y_true_svm = zheng_labels
y_pred_svm = zheng_model_svm.predict(zheng_data)

np.savez(cleandataDIR_intra + "svm_y_zheng.npz", y_true_svm, y_pred_svm)

accuracy_zheng_svm = accuracy_score(y_true_svm, y_pred_svm)
f1_zheng_svm = f1_score(y_true_svm, y_pred_svm, average='macro')

print("Accuracy SVM: ", accuracy_zheng_svm)
print("F1 score SVM: ", f1_zheng_svm)

#%% linear SVM zheng

y_true_linearsvm = zheng_labels
y_pred_linearsvm = zheng_model_linearsvm.predict(zheng_data)

np.savez(cleandataDIR_intra + "linearsvm_y_zheng.npz", y_true_linearsvm, y_pred_linearsvm)

accuracy_zheng_linearsvm = accuracy_score(y_true_linearsvm, y_pred_linearsvm)
f1_zheng_linearsvm = f1_score(y_true_linearsvm, y_pred_linearsvm, average='macro')

print("Accuracy linearsvm: ", accuracy_zheng_linearsvm)
print("F1 score linearsvm: ", f1_zheng_linearsvm)

#%% CNN zheng

y_true_cnn_raw = zheng_labels_onehot
y_true_cnn = np.argmax(zheng_labels_onehot, axis=-1)
y_pred_prob_cnn = zheng_model_cnn.predict(zheng_data)
y_pred_cnn = np.argmax(y_pred_prob_cnn, axis=-1)

np.savez(cleandataDIR_intra + "cnn_y_zheng.npz", y_true_cnn, y_pred_cnn, y_pred_prob_cnn)

accuracy_zheng_cnn = accuracy_score(y_true_cnn, y_pred_cnn)
f1_zheng_cnn = f1_score(y_true_cnn, y_pred_cnn, average='macro')

print("Accuracy cnn: ", accuracy_zheng_cnn)
print("F1 score cnn: ", f1_zheng_cnn)

#%% lstm zheng

y_true_lstm_raw = zheng_labels_onehot
y_true_lstm = np.argmax(zheng_labels_onehot, axis=-1)
y_pred_prob_lstm = zheng_model_lstm.predict(zheng_sentences)
y_pred_lstm = np.argmax(y_pred_prob_lstm, axis=-1)

np.savez(cleandataDIR_intra + "lstm_y_zheng.npz", y_true_lstm, y_pred_lstm, y_pred_prob_lstm)

accuracy_zheng_lstm = accuracy_score(y_true_lstm, y_pred_lstm)
f1_zheng_lstm = f1_score(y_true_lstm, y_pred_lstm, average='macro')

print("Accuracy lstm: ", accuracy_zheng_lstm)
print("F1 score lstm: ", f1_zheng_lstm)

#%% hybrid zheng

y_true_hybrid_raw = zheng_labels_onehot
y_true_hybrid = np.argmax(zheng_labels_onehot, axis=-1)
y_pred_prob_hybrid = zheng_model_hybrid.predict(zheng_sentences)
y_pred_hybrid = np.argmax(y_pred_prob_hybrid, axis=-1)

np.savez(cleandataDIR_intra + "hybrid_y_zheng.npz", y_true_hybrid, y_pred_hybrid, y_pred_prob_hybrid)

accuracy_zheng_hybrid = accuracy_score(y_true_hybrid, y_pred_hybrid)
f1_zheng_hybrid = f1_score(y_true_hybrid, y_pred_hybrid, average='macro')

print("Accuracy hybrid: ", accuracy_zheng_hybrid)
print("F1 score hybrid: ", f1_zheng_hybrid)

#%% ROC of deep learning models

zheng_data, zheng_sentences, zheng_labels, zheng_labels_onehot = get_data(
    "zheng", cleandataDIR_intra)

with np.load(cleandataDIR_intra + "cnn_y_zheng.npz", allow_pickle=True) as data:
    y_true_cnn = data["arr_0"]
    y_pred_cnn = data["arr_1"]
    y_pred_prob_cnn = data["arr_2"]

with np.load(cleandataDIR_intra + "lstm_y_zheng.npz", allow_pickle=True) as data:
    y_true_lstm = data["arr_0"]
    y_pred_lstm = data["arr_1"]
    y_pred_prob_lstm = data["arr_2"]

with np.load(cleandataDIR_intra + "hybrid_y_zheng.npz", allow_pickle=True) as data:
    y_true_hybrid = data["arr_0"]
    y_pred_hybrid = data["arr_1"]
    y_pred_prob_hybrid = data["arr_2"]

fpr_cnn = {}
tpr_cnn = {}
roc_auc_cnn = {}

fpr_lstm = {}
tpr_lstm = {}
roc_auc_lstm = {}

fpr_hybrid = {}
tpr_hybrid = {}
roc_auc_hybrid = {}

for i in range(zheng_labels_onehot.shape[1]):
    fpr_cnn[i], tpr_cnn[i], _ = roc_curve(zheng_labels_onehot[:,i], y_pred_prob_cnn[:,i])
    roc_auc_cnn[i] = auc(fpr_cnn[i], tpr_cnn[i])

for i in range(zheng_labels_onehot.shape[1]):
    fpr_lstm[i], tpr_lstm[i], _ = roc_curve(zheng_labels_onehot[:,i], y_pred_prob_lstm[:,i])
    roc_auc_lstm[i] = auc(fpr_lstm[i], tpr_lstm[i])

for i in range(zheng_labels_onehot.shape[1]):
    fpr_hybrid[i], tpr_hybrid[i], _ = roc_curve(zheng_labels_onehot[:,i], y_pred_prob_hybrid[:,i])
    roc_auc_hybrid[i] = auc(fpr_hybrid[i], tpr_hybrid[i])


# Compute micro-average ROC curve and ROC area
fpr_cnn["micro"], tpr_cnn["micro"], _ = roc_curve(zheng_labels_onehot.ravel(), y_pred_prob_cnn.ravel())
roc_auc_cnn["micro"] = auc(fpr_cnn["micro"], tpr_cnn["micro"])

fpr_lstm["micro"], tpr_lstm["micro"], _ = roc_curve(zheng_labels_onehot.ravel(), y_pred_prob_lstm.ravel())
roc_auc_lstm["micro"] = auc(fpr_lstm["micro"], tpr_lstm["micro"])

fpr_hybrid["micro"], tpr_hybrid["micro"], _ = roc_curve(zheng_labels_onehot.ravel(), y_pred_prob_hybrid.ravel())
roc_auc_hybrid["micro"] = auc(fpr_hybrid["micro"], tpr_hybrid["micro"])


# Compute macro-average ROC curve and ROC area
all_fpr_cnn = np.unique(np.concatenate([fpr_cnn[i] for i in range(zheng_labels_onehot.shape[1])]))
all_fpr_lstm = np.unique(np.concatenate([fpr_lstm[i] for i in range(zheng_labels_onehot.shape[1])]))
all_fpr_hybrid = np.unique(np.concatenate([fpr_hybrid[i] for i in range(zheng_labels_onehot.shape[1])]))

mean_tpr_cnn = np.zeros_like(all_fpr_cnn)
for i in range(zheng_labels_onehot.shape[1]):
    mean_tpr_cnn += interp(all_fpr_cnn, fpr_cnn[i], tpr_cnn[i])

mean_tpr_lstm = np.zeros_like(all_fpr_lstm)
for i in range(zheng_labels_onehot.shape[1]):
    mean_tpr_lstm += interp(all_fpr_lstm, fpr_lstm[i], tpr_lstm[i])

mean_tpr_hybrid = np.zeros_like(all_fpr_hybrid)
for i in range(zheng_labels_onehot.shape[1]):
    mean_tpr_hybrid += interp(all_fpr_hybrid, fpr_hybrid[i], tpr_hybrid[i])

mean_tpr_cnn /= zheng_labels_onehot.shape[1]
mean_tpr_lstm /= zheng_labels_onehot.shape[1]
mean_tpr_hybrid /= zheng_labels_onehot.shape[1]

fpr_cnn["macro"] = all_fpr_cnn
tpr_cnn["macro"] = mean_tpr_cnn
roc_auc_cnn["macro"] = auc(fpr_cnn["macro"], tpr_cnn["macro"])

fpr_lstm["macro"] = all_fpr_lstm
tpr_lstm["macro"] = mean_tpr_lstm
roc_auc_lstm["macro"] = auc(fpr_lstm["macro"], tpr_lstm["macro"])

fpr_hybrid["macro"] = all_fpr_hybrid
tpr_hybrid["macro"] = mean_tpr_hybrid
roc_auc_hybrid["macro"] = auc(fpr_hybrid["macro"], tpr_hybrid["macro"])


# Plot ROC curves
plt.figure(1)

plt.plot(fpr_cnn["macro"], tpr_cnn["macro"],
         label='CNN (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_cnn["macro"]),
         linewidth=2)

plt.plot(fpr_lstm["macro"], tpr_lstm["macro"],
         label='LSTM (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_lstm["macro"]),
         linewidth=2)

plt.plot(fpr_hybrid["macro"], tpr_hybrid["macro"],
         label='Hybrid (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_hybrid["macro"]),
         linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC curve of zheng dataset')
plt.legend(loc="lower right")

# Save plot
plt.savefig('Figures/ROC zheng.png', dpi=300)
plt.show()

#%% ROC plot zheng per class

x = [0, 1, 2, 3, 4]
y = [xx*xx for xx in x]

fig = plt.figure(figsize=(14,8))
ax  = fig.add_subplot(111)

ax.set_position([0.1,0.1,0.5,0.8])

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i in range(zheng_labels_onehot.shape[1]):
    plt.plot(fpr_hybrid[i], tpr_hybrid[i], lw=2,
             label='C{0} (AUC = {1:0.4f})'
             ''.format(i, roc_auc_hybrid[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(bbox_to_anchor=(1.01,1))

# Save plot
plt.savefig('Figures/cell level ROC zheng.png', dpi=300)
plt.show()

#%% Confusion matrix hybrid model zheng

cm = confusion_matrix(y_true_hybrid, y_pred_hybrid)

fig = plt.figure(figsize=(28, 26))
ax = plt.subplot()
sns.set(font_scale=3.0)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g', cmap='Blues')

class_names = np.sort(np.unique(zheng_labels))

class_names = []

for i in range(zheng_labels_onehot.shape[1]):
    class_names.append("C" + str(i+1))

ax.xaxis.set_label_position('bottom')
#plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 20)
ax.set_xlabel('Predicted', fontsize=36)
ax.xaxis.tick_bottom()

ax.yaxis.set_ticklabels(class_names, fontsize = 20)
ax.set_ylabel('True', fontsize=36)
plt.yticks(rotation=0)

plt.savefig('Figures/CM zheng.png', dpi=300)
plt.show()





#%%####################
##### Baron human #####
#######################

baron_human_performance, baron_human_model_cnn, baron_human_model_lstm, baron_human_model_hybrid, baron_human_model_svm, baron_human_model_linearsvm = get_performance_and_model(
    "baron_human", cleandataDIR_intra, 'intra')

baron_human_data, baron_human_sentences, baron_human_labels, baron_human_labels_onehot = get_data(
    "baron_human", cleandataDIR_intra)

#%% SVM baron_human

with np.load(cleandataDIR_intra + "intra_svm_pred_test_baron_human.npz", allow_pickle=True) as data:
    y_pred_svm = data["arr_1"]

y_true_svm = baron_human_labels
y_pred_svm = baron_human_model_svm.predict(baron_human_data)

np.savez(cleandataDIR_intra + "svm_y_baron_human.npz", y_true_svm, y_pred_svm)

accuracy_baron_human_svm = accuracy_score(y_true_svm, y_pred_svm)
f1_baron_human_svm = f1_score(y_true_svm, y_pred_svm, average='macro')

print("Accuracy SVM: ", accuracy_baron_human_svm)
print("F1 score SVM: ", f1_baron_human_svm)

#%% linear SVM baron_human

with np.load(cleandataDIR_intra + "intra_linear_svm_pred_test_baron_human.npz", allow_pickle=True) as data:
    y_pred_linearsvm = data["arr_1"]

y_true_linearsvm = baron_human_labels
y_pred_linearsvm = baron_human_model_linearsvm.predict(baron_human_data)

np.savez(cleandataDIR_intra + "linearsvm_y_baron_human.npz", y_true_linearsvm, y_pred_linearsvm)

accuracy_baron_human_linearsvm = accuracy_score(y_true_linearsvm, y_pred_linearsvm)
f1_baron_human_linearsvm = f1_score(y_true_linearsvm, y_pred_linearsvm, average='macro')

print("Accuracy linearsvm: ", accuracy_baron_human_linearsvm)
print("F1 score linearsvm: ", f1_baron_human_linearsvm)

#%% CNN baron_human

y_true_cnn_raw = baron_human_labels_onehot
y_true_cnn = np.argmax(baron_human_labels_onehot, axis=-1)
y_pred_prob_cnn = baron_human_model_cnn.predict(baron_human_data)
y_pred_cnn = np.argmax(y_pred_prob_cnn, axis=-1)

np.savez(cleandataDIR_intra + "cnn_y_baron_human.npz", y_true_cnn, y_pred_cnn, y_pred_prob_cnn)

accuracy_baron_human_cnn = accuracy_score(y_true_cnn, y_pred_cnn)
f1_baron_human_cnn = f1_score(y_true_cnn, y_pred_cnn, average='macro')

print("Accuracy cnn: ", accuracy_baron_human_cnn)
print("F1 score cnn: ", f1_baron_human_cnn)

#%% lstm baron_human

y_true_lstm_raw = baron_human_labels_onehot
y_true_lstm = np.argmax(baron_human_labels_onehot, axis=-1)
y_pred_prob_lstm = baron_human_model_lstm.predict(baron_human_sentences)
y_pred_lstm = np.argmax(y_pred_prob_lstm, axis=-1)

np.savez(cleandataDIR_intra + "lstm_y_baron_human.npz", y_true_lstm, y_pred_lstm, y_pred_prob_lstm)

accuracy_baron_human_lstm = accuracy_score(y_true_lstm, y_pred_lstm)
f1_baron_human_lstm = f1_score(y_true_lstm, y_pred_lstm, average='macro')

print("Accuracy lstm: ", accuracy_baron_human_lstm)
print("F1 score lstm: ", f1_baron_human_lstm)

#%% hybrid baron_human

y_true_hybrid_raw = baron_human_labels_onehot
y_true_hybrid = np.argmax(baron_human_labels_onehot, axis=-1)
y_pred_prob_hybrid = baron_human_model_hybrid.predict(baron_human_sentences)
y_pred_hybrid = np.argmax(y_pred_prob_hybrid, axis=-1)

np.savez(cleandataDIR_intra + "hybrid_y_baron_human.npz", y_true_hybrid, y_pred_hybrid, y_pred_prob_hybrid)

accuracy_baron_human_hybrid = accuracy_score(y_true_hybrid, y_pred_hybrid)
f1_baron_human_hybrid = f1_score(y_true_hybrid, y_pred_hybrid, average='macro')

print("Accuracy hybrid: ", accuracy_baron_human_hybrid)
print("F1 score hybrid: ", f1_baron_human_hybrid)

#%% ROC of deep learning models

baron_human_data, baron_human_sentences, baron_human_labels, baron_human_labels_onehot = get_data(
    "baron_human", cleandataDIR_intra)

with np.load(cleandataDIR_intra + "cnn_y_baron_human.npz", allow_pickle=True) as data:
    y_true_cnn = data["arr_0"]
    y_pred_cnn = data["arr_1"]
    y_pred_prob_cnn = data["arr_2"]

with np.load(cleandataDIR_intra + "lstm_y_baron_human.npz", allow_pickle=True) as data:
    y_true_lstm = data["arr_0"]
    y_pred_lstm = data["arr_1"]
    y_pred_prob_lstm = data["arr_2"]

with np.load(cleandataDIR_intra + "hybrid_y_baron_human.npz", allow_pickle=True) as data:
    y_true_hybrid = data["arr_0"]
    y_pred_hybrid = data["arr_1"]
    y_pred_prob_hybrid = data["arr_2"]

fpr_cnn = {}
tpr_cnn = {}
roc_auc_cnn = {}

fpr_lstm = {}
tpr_lstm = {}
roc_auc_lstm = {}

fpr_hybrid = {}
tpr_hybrid = {}
roc_auc_hybrid = {}

for i in range(baron_human_labels_onehot.shape[1]):
    fpr_cnn[i], tpr_cnn[i], _ = roc_curve(baron_human_labels_onehot[:,i], y_pred_prob_cnn[:,i])
    roc_auc_cnn[i] = auc(fpr_cnn[i], tpr_cnn[i])

for i in range(baron_human_labels_onehot.shape[1]):
    fpr_lstm[i], tpr_lstm[i], _ = roc_curve(baron_human_labels_onehot[:,i], y_pred_prob_lstm[:,i])
    roc_auc_lstm[i] = auc(fpr_lstm[i], tpr_lstm[i])

for i in range(baron_human_labels_onehot.shape[1]):
    fpr_hybrid[i], tpr_hybrid[i], _ = roc_curve(baron_human_labels_onehot[:,i], y_pred_prob_hybrid[:,i])
    roc_auc_hybrid[i] = auc(fpr_hybrid[i], tpr_hybrid[i])


# Compute micro-average ROC curve and ROC area
fpr_cnn["micro"], tpr_cnn["micro"], _ = roc_curve(baron_human_labels_onehot.ravel(), y_pred_prob_cnn.ravel())
roc_auc_cnn["micro"] = auc(fpr_cnn["micro"], tpr_cnn["micro"])

fpr_lstm["micro"], tpr_lstm["micro"], _ = roc_curve(baron_human_labels_onehot.ravel(), y_pred_prob_lstm.ravel())
roc_auc_lstm["micro"] = auc(fpr_lstm["micro"], tpr_lstm["micro"])

fpr_hybrid["micro"], tpr_hybrid["micro"], _ = roc_curve(baron_human_labels_onehot.ravel(), y_pred_prob_hybrid.ravel())
roc_auc_hybrid["micro"] = auc(fpr_hybrid["micro"], tpr_hybrid["micro"])


# Compute macro-average ROC curve and ROC area
all_fpr_cnn = np.unique(np.concatenate([fpr_cnn[i] for i in range(baron_human_labels_onehot.shape[1])]))
all_fpr_lstm = np.unique(np.concatenate([fpr_lstm[i] for i in range(baron_human_labels_onehot.shape[1])]))
all_fpr_hybrid = np.unique(np.concatenate([fpr_hybrid[i] for i in range(baron_human_labels_onehot.shape[1])]))

mean_tpr_cnn = np.zeros_like(all_fpr_cnn)
for i in range(baron_human_labels_onehot.shape[1]):
    mean_tpr_cnn += interp(all_fpr_cnn, fpr_cnn[i], tpr_cnn[i])

mean_tpr_lstm = np.zeros_like(all_fpr_lstm)
for i in range(baron_human_labels_onehot.shape[1]):
    mean_tpr_lstm += interp(all_fpr_lstm, fpr_lstm[i], tpr_lstm[i])

mean_tpr_hybrid = np.zeros_like(all_fpr_hybrid)
for i in range(baron_human_labels_onehot.shape[1]):
    mean_tpr_hybrid += interp(all_fpr_hybrid, fpr_hybrid[i], tpr_hybrid[i])

mean_tpr_cnn /= baron_human_labels_onehot.shape[1]
mean_tpr_lstm /= baron_human_labels_onehot.shape[1]
mean_tpr_hybrid /= baron_human_labels_onehot.shape[1]

fpr_cnn["macro"] = all_fpr_cnn
tpr_cnn["macro"] = mean_tpr_cnn
roc_auc_cnn["macro"] = auc(fpr_cnn["macro"], tpr_cnn["macro"])

fpr_lstm["macro"] = all_fpr_lstm
tpr_lstm["macro"] = mean_tpr_lstm
roc_auc_lstm["macro"] = auc(fpr_lstm["macro"], tpr_lstm["macro"])

fpr_hybrid["macro"] = all_fpr_hybrid
tpr_hybrid["macro"] = mean_tpr_hybrid
roc_auc_hybrid["macro"] = auc(fpr_hybrid["macro"], tpr_hybrid["macro"])


# Plot ROC curves
plt.figure(1)

plt.plot(fpr_cnn["macro"], tpr_cnn["macro"],
         label='CNN (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_cnn["macro"]),
         linewidth=2)

plt.plot(fpr_lstm["macro"], tpr_lstm["macro"],
         label='LSTM (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_lstm["macro"]),
         linewidth=2)

plt.plot(fpr_hybrid["macro"], tpr_hybrid["macro"],
         label='Hybrid (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_hybrid["macro"]),
         linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC curve of baron human dataset')
plt.legend(loc="lower right")

# Save plot
plt.savefig('Figures/ROC baron_human.png', dpi=300)
plt.show()

#%% ROC plot baron_human per class

x = [0, 1, 2, 3, 4]
y = [xx*xx for xx in x]

fig = plt.figure(figsize=(14,8))
ax  = fig.add_subplot(111)

ax.set_position([0.1,0.1,0.5,0.8])

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i in range(baron_human_labels_onehot.shape[1]):
    plt.plot(fpr_hybrid[i], tpr_hybrid[i], lw=2,
             label='C{0} (AUC = {1:0.4f})'
             ''.format(i, roc_auc_hybrid[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(bbox_to_anchor=(1.01,1))

# Save plot
plt.savefig('Figures/cell level ROC baron_human.png', dpi=300)
plt.show()

#%% Confusion matrix hybrid model baron_human

cm = confusion_matrix(y_true_hybrid, y_pred_hybrid)

fig = plt.figure(figsize=(28, 26))
ax = plt.subplot()
sns.set(font_scale=3.0)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g', cmap='Blues')

#class_names = np.sort(np.unique(baron_human_labels))

class_names = []

for i in range(baron_human_labels_onehot.shape[1]):
    class_names.append("C" + str(i+1))

ax.xaxis.set_label_position('bottom')
#plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 20)
ax.set_xlabel('Predicted', fontsize=36)
ax.xaxis.tick_bottom()

ax.yaxis.set_ticklabels(class_names, fontsize = 20)
ax.set_ylabel('True', fontsize=36)
plt.yticks(rotation=0)

plt.savefig('Figures/CM baron_human.png', dpi=300)
plt.show()





#%%##############
##### Inter #####
#################

inter_performance, inter_model_cnn, inter_model_lstm, inter_model_hybrid, inter_model_svm, inter_model_linearsvm = get_performance_and_model(
    "inter_pbmc", cleandataDIR_inter, analysis='inter')

inter_data, inter_sentences, inter_labels, inter_labels_onehot = get_data(
    "pbmc", cleandataDIR_inter)

#%% SVM inter

with np.load(cleandataDIR_inter + "inter_svm_pred_test.npz", allow_pickle=True) as data:
    y_pred_svm = data["arr_1"]

y_true_svm = inter_labels
y_pred_svm = inter_model_svm.predict(inter_data)

np.savez(cleandataDIR_inter + "svm_y_inter.npz", y_true_svm, y_pred_svm)

accuracy_inter_svm = accuracy_score(y_true_svm, y_pred_svm)
f1_inter_svm = f1_score(y_true_svm, y_pred_svm, average='macro')

print("Accuracy SVM: ", accuracy_inter_svm)
print("F1 score SVM: ", f1_inter_svm)

#%% linear SVM inter

with np.load(cleandataDIR_inter + "inter_linear_svm_pred_test.npz", allow_pickle=True) as data:
    y_pred_linearsvm = data["arr_1"]

y_true_linearsvm = inter_labels
y_pred_linearsvm = inter_model_linearsvm.predict(inter_data)

np.savez(cleandataDIR_inter + "linearsvm_y_inter.npz", y_true_linearsvm, y_pred_linearsvm)

accuracy_inter_linearsvm = accuracy_score(y_true_linearsvm, y_pred_linearsvm)
f1_inter_linearsvm = f1_score(y_true_linearsvm, y_pred_linearsvm, average='macro')

print("Accuracy linearsvm: ", accuracy_inter_linearsvm)
print("F1 score linearsvm: ", f1_inter_linearsvm)

#%% CNN inter

y_true_cnn_raw = inter_labels_onehot
y_true_cnn = np.argmax(inter_labels_onehot, axis=-1)
y_pred_prob_cnn = inter_model_cnn.predict(inter_data)
y_pred_cnn = np.argmax(y_pred_prob_cnn, axis=-1)

np.savez(cleandataDIR_inter + "cnn_y_inter.npz", y_true_cnn, y_pred_cnn, y_pred_prob_cnn)

accuracy_inter_cnn = accuracy_score(y_true_cnn, y_pred_cnn)
f1_inter_cnn = f1_score(y_true_cnn, y_pred_cnn, average='macro')

print("Accuracy cnn: ", accuracy_inter_cnn)
print("F1 score cnn: ", f1_inter_cnn)

#%% lstm inter

y_true_lstm_raw = inter_labels_onehot
y_true_lstm = np.argmax(inter_labels_onehot, axis=-1)
y_pred_prob_lstm = inter_model_lstm.predict(inter_sentences)
y_pred_lstm = np.argmax(y_pred_prob_lstm, axis=-1)

np.savez(cleandataDIR_inter + "lstm_y_inter.npz", y_true_lstm, y_pred_lstm, y_pred_prob_lstm)

accuracy_inter_lstm = accuracy_score(y_true_lstm, y_pred_lstm)
f1_inter_lstm = f1_score(y_true_lstm, y_pred_lstm, average='macro')

print("Accuracy lstm: ", accuracy_inter_lstm)
print("F1 score lstm: ", f1_inter_lstm)

#%% hybrid inter

y_true_hybrid_raw = inter_labels_onehot
y_true_hybrid = np.argmax(inter_labels_onehot, axis=-1)
y_pred_prob_hybrid = inter_model_hybrid.predict(inter_sentences)
y_pred_hybrid = np.argmax(y_pred_prob_hybrid, axis=-1)

np.savez(cleandataDIR_inter + "hybrid_y_inter.npz", y_true_hybrid, y_pred_hybrid, y_pred_prob_hybrid)

accuracy_inter_hybrid = accuracy_score(y_true_hybrid, y_pred_hybrid)
f1_inter_hybrid = f1_score(y_true_hybrid, y_pred_hybrid, average='macro')

print("Accuracy hybrid: ", accuracy_inter_hybrid)
print("F1 score hybrid: ", f1_inter_hybrid)

#%% ROC of deep learning models

inter_data, inter_sentences, inter_labels, inter_labels_onehot = get_data(
    "pbmc", cleandataDIR_inter)

with np.load(cleandataDIR_inter + "cnn_y_inter.npz", allow_pickle=True) as data:
    y_true_cnn = data["arr_0"]
    y_pred_cnn = data["arr_1"]
    y_pred_prob_cnn = data["arr_2"]

with np.load(cleandataDIR_inter + "lstm_y_inter.npz", allow_pickle=True) as data:
    y_true_lstm = data["arr_0"]
    y_pred_lstm = data["arr_1"]
    y_pred_prob_lstm = data["arr_2"]

with np.load(cleandataDIR_inter + "hybrid_y_inter.npz", allow_pickle=True) as data:
    y_true_hybrid = data["arr_0"]
    y_pred_hybrid = data["arr_1"]
    y_pred_prob_hybrid = data["arr_2"]

fpr_cnn = {}
tpr_cnn = {}
roc_auc_cnn = {}

fpr_lstm = {}
tpr_lstm = {}
roc_auc_lstm = {}

fpr_hybrid = {}
tpr_hybrid = {}
roc_auc_hybrid = {}

for i in range(inter_labels_onehot.shape[1]):
    fpr_cnn[i], tpr_cnn[i], _ = roc_curve(inter_labels_onehot[:,i], y_pred_prob_cnn[:,i])
    roc_auc_cnn[i] = auc(fpr_cnn[i], tpr_cnn[i])

for i in range(inter_labels_onehot.shape[1]):
    fpr_lstm[i], tpr_lstm[i], _ = roc_curve(inter_labels_onehot[:,i], y_pred_prob_lstm[:,i])
    roc_auc_lstm[i] = auc(fpr_lstm[i], tpr_lstm[i])

for i in range(inter_labels_onehot.shape[1]):
    fpr_hybrid[i], tpr_hybrid[i], _ = roc_curve(inter_labels_onehot[:,i], y_pred_prob_hybrid[:,i])
    roc_auc_hybrid[i] = auc(fpr_hybrid[i], tpr_hybrid[i])


# Compute micro-average ROC curve and ROC area
fpr_cnn["micro"], tpr_cnn["micro"], _ = roc_curve(inter_labels_onehot.ravel(), y_pred_prob_cnn.ravel())
roc_auc_cnn["micro"] = auc(fpr_cnn["micro"], tpr_cnn["micro"])

fpr_lstm["micro"], tpr_lstm["micro"], _ = roc_curve(inter_labels_onehot.ravel(), y_pred_prob_lstm.ravel())
roc_auc_lstm["micro"] = auc(fpr_lstm["micro"], tpr_lstm["micro"])

fpr_hybrid["micro"], tpr_hybrid["micro"], _ = roc_curve(inter_labels_onehot.ravel(), y_pred_prob_hybrid.ravel())
roc_auc_hybrid["micro"] = auc(fpr_hybrid["micro"], tpr_hybrid["micro"])


# Compute macro-average ROC curve and ROC area
all_fpr_cnn = np.unique(np.concatenate([fpr_cnn[i] for i in range(inter_labels_onehot.shape[1])]))
all_fpr_lstm = np.unique(np.concatenate([fpr_lstm[i] for i in range(inter_labels_onehot.shape[1])]))
all_fpr_hybrid = np.unique(np.concatenate([fpr_hybrid[i] for i in range(inter_labels_onehot.shape[1])]))

mean_tpr_cnn = np.zeros_like(all_fpr_cnn)
for i in range(inter_labels_onehot.shape[1]):
    mean_tpr_cnn += interp(all_fpr_cnn, fpr_cnn[i], tpr_cnn[i])

mean_tpr_lstm = np.zeros_like(all_fpr_lstm)
for i in range(inter_labels_onehot.shape[1]):
    mean_tpr_lstm += interp(all_fpr_lstm, fpr_lstm[i], tpr_lstm[i])

mean_tpr_hybrid = np.zeros_like(all_fpr_hybrid)
for i in range(inter_labels_onehot.shape[1]):
    mean_tpr_hybrid += interp(all_fpr_hybrid, fpr_hybrid[i], tpr_hybrid[i])

mean_tpr_cnn /= inter_labels_onehot.shape[1]
mean_tpr_lstm /= inter_labels_onehot.shape[1]
mean_tpr_hybrid /= inter_labels_onehot.shape[1]

fpr_cnn["macro"] = all_fpr_cnn
tpr_cnn["macro"] = mean_tpr_cnn
roc_auc_cnn["macro"] = auc(fpr_cnn["macro"], tpr_cnn["macro"])

fpr_lstm["macro"] = all_fpr_lstm
tpr_lstm["macro"] = mean_tpr_lstm
roc_auc_lstm["macro"] = auc(fpr_lstm["macro"], tpr_lstm["macro"])

fpr_hybrid["macro"] = all_fpr_hybrid
tpr_hybrid["macro"] = mean_tpr_hybrid
roc_auc_hybrid["macro"] = auc(fpr_hybrid["macro"], tpr_hybrid["macro"])


# Plot ROC curves
plt.figure(1)

plt.plot(fpr_cnn["macro"], tpr_cnn["macro"],
         label='CNN (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_cnn["macro"]),
         linewidth=2)

plt.plot(fpr_lstm["macro"], tpr_lstm["macro"],
         label='LSTM (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_lstm["macro"]),
         linewidth=2)

plt.plot(fpr_hybrid["macro"], tpr_hybrid["macro"],
         label='Hybrid (Macro-average AUC = {0:0.4f})'
               ''.format(roc_auc_hybrid["macro"]),
         linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC curve of inter dataset')
plt.legend(loc="lower right")

# Save plot
plt.savefig('Figures/ROC inter.png', dpi=300)
plt.show()

#%% ROC plot inter per class

x = [0, 1, 2, 3, 4]
y = [xx*xx for xx in x]

fig = plt.figure(figsize=(14,8))
ax  = fig.add_subplot(111)

ax.set_position([0.1,0.1,0.5,0.8])

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i in range(inter_labels_onehot.shape[1]):
    plt.plot(fpr_hybrid[i], tpr_hybrid[i], lw=2,
             label='C{0} (AUC = {1:0.4f})'
             ''.format(i, roc_auc_hybrid[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(bbox_to_anchor=(1.01,1))

# Save plot
plt.savefig('Figures/cell level ROC inter.png', dpi=300)
plt.show()

#%% Confusion matrix hybrid model inter

cm = confusion_matrix(y_true_hybrid, y_pred_hybrid)

fig = plt.figure(figsize=(28, 26))
ax = plt.subplot()
sns.set(font_scale=3.0)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g', cmap='Blues')

#class_names = np.sort(np.unique(inter_labels))

class_names = []

for i in range(inter_labels_onehot.shape[1]):
    class_names.append("C" + str(i+1))

ax.xaxis.set_label_position('bottom')
#plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 20)
ax.set_xlabel('Predicted', fontsize=36)
ax.xaxis.tick_bottom()

ax.yaxis.set_ticklabels(class_names, fontsize = 20)
ax.set_ylabel('True', fontsize=36)
plt.yticks(rotation=0)

plt.savefig('Figures/CM inter.png', dpi=300)
plt.show()