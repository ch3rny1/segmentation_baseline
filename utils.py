import numpy as np


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    eps = 0.0001
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f) + eps)
