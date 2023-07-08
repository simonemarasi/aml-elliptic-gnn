import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support, f1_score

def get_config():
    with open(os.path.join(os.getcwd(), "config.yaml"), "r") as config:
        args = AttributeDict(yaml.safe_load(config))
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
    return args

def accuracy(pred_y, y):
    """Calculate accuracy"""
    return ((pred_y == y).sum() / len(y)).item()

def compute_metrics(model, name, data, df):

  _, y_predicted = model((data.x, data.edge_index))[0].to("cpu").max(dim=1)
  data = data.to("cpu")

  prec_ill,rec_ill,f1_ill,_ = precision_recall_fscore_support(data.y[data.test_mask], y_predicted[data.test_mask], average='binary', pos_label=0)
  f1_micro = f1_score(data.y[data.test_mask], y_predicted[data.test_mask], average='micro')

  m = {'model': name, 'Precision': np.round(prec_ill,3), 'Recall': np.round(rec_ill,3), 'F1': np.round(f1_ill,3),
   'F1 Micro AVG':np.round(f1_micro,3)}

  return m

def plot_results(df):

    labels = df['model'].to_numpy()
    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    f1_micro = df['F1 Micro AVG'].to_numpy()

    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(x - width/2, precision, width, label='Precision',color='#83f27b')
    ax.bar(x + width/2, recall, width, label='Recall',color='#f27b83')
    ax.bar(x - (3/2)*width, f1, width, label='F1',color='#f2b37b')
    ax.bar(x + (3/2)*width, f1_micro, width, label='Micro AVG F1',color='#7b8bf2')

    ax.set_ylabel('value')
    ax.set_title('Metrics for illicit class')
    ax.set_xticks(np.arange(0,len(labels),1))
    ax.set_yticks(np.arange(0,1,0.05))
    ax.set_xticklabels(labels=labels)
    ax.legend(loc="lower left")

    plt.grid(True)
    plt.show()

class AttributeDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__