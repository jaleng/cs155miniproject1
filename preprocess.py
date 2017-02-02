import os.path
import numpy as np
import pkl_help
from pkl_help import read_make_pkl

def get_csv_data(filename):
    data = np.genfromtxt(filename, delimiter=",")
    return data

def remove_header_and_normalize(data):
    data = np.delete(data, 0, axis=0) # get ride of id column
    data = np.delete(data, 0, axis=1) # get ride of header
    abs_data = np.absolute(data)
    data_norm = np.divide(data, abs_data.max(axis=0), 
                          out=np.zeros_like(data), 
                          where=abs_data.max(axis=0)!=0) #divide by max of absolute value. If it's zero place a 0 there.
    return data_norm

train_2008 = read_make_pkl("saved_objs/train_2008.pkl",
                           lambda: get_csv_data("data/train_2008.csv"),
                           compress=True)

test_2008 = read_make_pkl("saved_objs/test_2008.pkl",
                          lambda: get_csv_data("data/test_2008.csv"),
                          compress=True)

test_2012 = read_make_pkl("saved_objs/test_2012.pkl",
                          lambda: get_csv_data("data/test_2012.csv"),
                          compress=True)

# Need to split number off to preserve validation set for ensemble/ final validation if we 
# want early stopping. We should choose a number of points to preserve.

pre_processed_train_2008 = read_make_pkl("saved_objs/pre_processed_train_2008.pkl",
                           lambda: remove_header_and_normalize(train_2008),
                           compress=True)

pre_processed_test_2008 =  read_make_pkl("saved_objs/pre_processed_test_2008.pkl",
                           lambda: remove_header_and_normalize(test_2008),
                           compress=True)

pre_processed_test_2012 = read_make_pkl("saved_objs/pre_processed_test_2012.pkl",
                           lambda: remove_header_and_normalize(test_2012),
                           compress=True)

# Debug
print("train_2008.shape = " + str(train_2008.shape))
print("test_2008.shape = " + str(test_2008.shape))
print("test_2012.shape = " + str(test_2012.shape))
