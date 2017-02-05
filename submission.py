# submission.py
import numpy as np
from pkl_help import get_pkl

def make_submission_2008(fn, predictions):
    # get 2008 ids
    ids = get_pkl("saved_objs/test_2008.pkl")[1:, 0:1]

    # If passed 1d array, change to column
    if (predictions.ndim == 1):
        predictions = predictions.reshape((-1, 1))
    # DEBUG
    print("ids.shape: " + str(ids.shape))
    print("predictions.shape: " + str(predictions.shape))
    np.savetxt(fn, np.concatenate((ids, predictions), axis=1),
               delimiter=',', header="id,PES1", comments='',
               fmt="%d,%d")

def make_submission_2012(fn, predictions):
    # get 2012 ids
    ids = get_pkl("saved_objs/test_2012.pkl")[1:, 0:1]
    np.savetxt(fn, np.concatenate((ids, predictions), axis=1),
               delimiter=',', header="id,PES1")
