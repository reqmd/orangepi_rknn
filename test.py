import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from utils.funcs import  load_yaml

def test(val_loader, params, model):
    params = load_yaml('config.yaml')
    loss_fn = nn.CrossEntropyLoss()
    y_preds = []
    y_trues = []
    test_acc = []

    model.eval()
    for X, y in val_loader:
        X = X.to(params['device'])
        y = y.to(params['device'])
        y_trues.extend(y.cpu().numpy())
        y_pred = model(X)
        val_loss = loss_fn(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        y_preds.extend(y_pred.cpu().numpy())
        acc = sum(y == y_pred) / len(y)
        test_acc.append(acc.cpu().detach().numpy())

    f1 = f1_score(y_trues, y_preds, average='weighted')
    return f1, np.round(np.mean(test_acc) * 100, 2)

