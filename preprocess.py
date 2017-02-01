import os.path
import numpy as np
import pkl_help
from pkl_help import read_make_pkl

def get_csv_data(filename):
    data = np.genfromtxt(filename, delimiter=",")
    return data

train_2008 = read_make_pkl("saved_objs/train_2008.pkl",
                           lambda: get_csv_data("data/train_2008.csv"),
                           compress=True)

test_2008 = read_make_pkl("saved_objs/test_2008.pkl",
                          lambda: get_csv_data("data/test_2008.csv"),
                          compress=True)

test_2012 = read_make_pkl("saved_objs/test_2012.pkl",
                          lambda: get_csv_data("data/test_2012.csv"),
                          compress=True)

# Debug
print("train_2008.shape = " + str(train_2008.shape))
print("test_2008.shape = " + str(test_2008.shape))
print("test_2012.shape = " + str(test_2012.shape))
