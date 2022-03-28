"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import numpy as np
import torch
import torch.nn as nn
import math
import dgl
from sklearn.metrics import roc_curve, auc

from train.metrics import AUC as accuracy

"""
    For GCNs
"""
def compute_roc_auc(labels, prediction, path_to_save=None, stage=None):

    fpr, tpr, thresholds = roc_curve(labels, prediction, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # print("TPR value: {}".format(tpr))
    # print("FPR value: {}".format(tpr))
    # print(thresholds)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ## locate the index of the largest g-mean
    #ix = np.argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    return roc_auc

def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):

        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0;
            sign_flip[sign_flip < 0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        scores_severity = []
        labels_lst = []
        for i in range(len(batch_scores)):
            score_value = float(batch_scores[i][1].item())
            lab_value = int(torch.argmax(batch_labels[i]).item())
            if i == 0:
                scores_severity = np.expand_dims(np.array(score_value), axis=0)
                labels_lst = np.expand_dims(np.array(lab_value), axis=0)
            else:
                scores_severity = np.concatenate((scores_severity, np.expand_dims(np.array(score_value), axis=0)),
                                                 axis=0)
                labels_lst = np.concatenate((labels_lst, np.expand_dims(np.array(lab_value), axis=0)), axis=0)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += compute_roc_auc(labels=labels_lst, prediction=scores_severity)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch, infer = False, infer5=False):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            scores_severity = []
            labels_lst = []
            for i in range(len(batch_scores)):
                score_value = float(batch_scores[i][1].item())
                lab_value = int(torch.argmax(batch_labels[i]).item())
                if i == 0:
                    scores_severity = np.expand_dims(np.array(score_value), axis=0)
                    labels_lst = np.expand_dims(np.array(lab_value), axis=0)
                else:
                    scores_severity = np.concatenate((scores_severity, np.expand_dims(np.array(score_value), axis=0)),
                                                     axis=0)
                    labels_lst = np.concatenate((labels_lst, np.expand_dims(np.array(lab_value), axis=0)), axis=0)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += compute_roc_auc(labels=labels_lst, prediction=scores_severity)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc


"""
    For WL-GNNs
"""


def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        scores_severity=[]
        labels_lst=[]
        scores = model.forward(x_with_node_feat)
        for i in range(len(scores)):
            score_value = float(scores[i][1].item())
            lab_value = int(torch.argmax(labels[i]).item())
            if i == 0:
                scores_severity = np.expand_dims(np.array(score_value), axis=0)
                labels_lst = np.expand_dims(np.array(lab_value), axis=0)
            else:
                scores_severity = np.concatenate((scores_severity, np.expand_dims(np.array(score_value), axis=0)),
                                                 axis=0)
                labels_lst = np.concatenate((labels_lst, np.expand_dims(np.array(lab_value), axis=0)), axis=0)
        loss = model.loss(scores, labels)
        loss.backward()

        if not (iter % batch_size):
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores_severity, labels_lst)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_dense(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)

            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc