import torch
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
)
from data import create_dataloader_new
from data.process import get_processing_model

def validate(model, opt):

    opt = get_processing_model(opt)

    data_loader = create_dataloader_new(opt)
    y_true, y_pred = [], []
    # with torch.no_grad():
    i = 0
    for img, label in data_loader:
        i += 1
        print("batch number {}/{}".format(i, len(data_loader)), end="\r")
        in_tens = img.cuda()
        # label = label.cuda()
        try:
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
        except Exception as e:
            print(e)
            continue
        del in_tens

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred
