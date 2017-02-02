import os.path
import numpy as np
import pkl_help
from pkl_help import read_make_pkl

def get_csv_data(filename):
    data = np.genfromtxt(filename, delimiter=",")
    return data

def remove_header_and_normalize(data):
    data = np.delete(data, 0, axis=0)
    data = np.delete(data, 0, axis=1)
    abs_data = np.absolute(data)
    data_norm = np.divide(data, abs_data.max(axis=0), 
                          out=np.zeros_like(data), 
                          where=abs_data.max(axis=0)!=0)
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

pre_processed_train_2008 = read_make_pkl("saved_objs/pre_processed_train_2008.pkl",
                           lambda: remove_header_and_normalize(train_2008),
                           compress=True)

pre_processed_test_2008 =  read_make_pkl("saved_objs/pre_processed_test_2008.pkl",
                           lambda: remove_header_and_normalize(test_2008),
                           compress=True)

pre_processed_test_2012 = read_make_pkl("saved_objs/pre_processed_test_2012.pkl",
                           lambda: remove_header_and_normalize(test_2012),
                           compress=True)
