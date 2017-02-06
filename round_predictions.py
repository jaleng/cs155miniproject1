# round_predictions.py
import numpy as np

def round_to_1_or_2(a):
    if (a[0] > 1.5):
        return 2
    else:
        return 1

def round_predictions(Y):
    preds = Y.reshape(-1, 1)
    return np.apply_along_axis(round_to_1_or_2, 1, preds)
