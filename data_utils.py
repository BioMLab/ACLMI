import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sk_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score


def metrics1(y_true, y_score):
    auc = sk_metrics.roc_auc_score(y_true, y_score)
    precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_score)
    au_prc = sk_metrics.auc(recall, precision)
    y_pred = [0 if i < 0.5 else 1 for i in y_score]
    acc = sk_metrics.accuracy_score(y_true, y_pred)
    pre = sk_metrics.precision_score(y_true, y_pred)
    rec = sk_metrics.recall_score(y_true, y_pred)
    f1 = sk_metrics.f1_score(y_true, y_pred)
    return {'auc': auc, 'aupr': au_prc, 'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}


def metrics2(targets, outputs):
    auc = roc_auc_score(targets, outputs)
    aupr = average_precision_score(targets, outputs)
    binary_outputs = np.where(np.array(outputs) > 0.5, 1, 0)
    accuracy = accuracy_score(targets, binary_outputs)
    recall = recall_score(targets, binary_outputs)
    precision = precision_score(targets, binary_outputs)
    f1 = f1_score(targets, binary_outputs)
    return {'AUC':auc, 'AUPR':aupr, 'Acc':accuracy, 'Rec':recall, 'Pre':precision, 'F1':f1}


def metrics_util(outputs, targets):
    auc = roc_auc_score(targets, outputs)
    auprc = average_precision_score(targets, outputs)
    preds = (outputs > 0.5).astype(int)
    acc = accuracy_score(targets, preds)
    pre = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    return {'AUC':auc, 'AUPR':auprc, 'Acc':acc, 'Pre':pre, 'Rec':rec, 'F1':f1}


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features1, features2, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features1.is_cuda
                  else torch.device('cpu'))

        features = torch.cat((features1,features2),0)
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        batch = features1.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            mask = torch.eye(batch, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)
        mask = torch.cat((mask,mask), 1)
        mask = torch.cat((mask, mask), 0)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask).to(exp_logits.device) - torch.eye(batch_size).to(exp_logits.device)
        positives_mask = (mask * logits_mask).to(logits_mask.device)
        negatives_mask = (1. - mask).to(logits_mask.device)

        num_positives_per_row = torch.sum(positives_mask, axis=1).to(logits_mask.device)
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
