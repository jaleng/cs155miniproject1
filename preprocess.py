import os.path
import numpy as np
import pkl_help
from pkl_help import read_make_pkl
from sklearn import preprocessing
import preprocess_help as ph

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
##################################################################################
X_train_2008 = read_make_pkl("saved_objs/X_train_2008.pkl",
                           lambda: ph.remove_header_and_normalize_train(train_2008),
                           compress=True)
Y_train_2008 = read_make_pkl("saved_objs/Y_train_2008.pkl",
                           lambda: ph.grab_train_Y(train_2008),
                           compress=True)
X_ver_2008 = read_make_pkl("saved_objs/X_ver_2008.pkl",
                           lambda: ph.remove_header_and_normalize_ver(train_2008),
                           compress=True)
Y_ver_2008 = read_make_pkl("saved_objs/Y_ver_2008.pkl",
                           lambda: ph.grab_ver_Y(train_2008),
                           compress=True)
X_test_2008 =  read_make_pkl("saved_objs/X_test_2008.pkl",
                           lambda: ph.remove_header_and_normalize_test(test_2008),
                           compress=True)
X_test_2012 = read_make_pkl("saved_objs/X_test_2012.pkl",
                           lambda: ph.remove_header_and_normalize_test(test_2012),
                           compress=True)
##################################################################################

# Debug
print("train_2008.shape = " + str(train_2008.shape))
print("test_2008.shape = " + str(test_2008.shape))
print("test_2012.shape = " + str(test_2012.shape))

print("X_train_2008.shape = " + str(X_train_2008.shape))
print("Y_train_2008.shape = " + str(Y_train_2008.shape))
print("X_ver_2008.shape = " + str(X_ver_2008.shape))
print("Y_ver_2008.shape = " + str(Y_ver_2008.shape))
print("X_test_2008.shape = " + str(X_test_2008.shape))
print("X_test_2012.shape = " + str(X_test_2012.shape))

