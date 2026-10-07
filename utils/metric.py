import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.counter = 0

def accuracy(output,label):
    y_pred_tag = torch.softmax(output, dim=1)
    # print(y_pred_tag.shape)
    output_label = torch.argmax(y_pred_tag, dim=1).detach().cpu()
    if len(label.shape) > 1:
        true_args = torch.argmax(label, dim=1).detach().cpu()
    else:
        true_args = label.detach().cpu()
    return (output_label == true_args).float().sum().numpy()

def compute_top_n_accuracy(preds, truths, n):
    truths = truths.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = np.argsort(truths, axis=1)[:,-1]
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]

def cf_matrix(truths, preds, n=2000):
    conf_mat = confusion_matrix(truths, preds)
    plt.figure(figsize=(10, 10))
    labels = np.arange(n)
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Truth') 
    plt.show()
