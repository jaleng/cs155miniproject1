"""
Helper functions to deal with preprocessing
"""
from sklearn import preprocessing
import numpy as np

N = 1000 #Save for verification.
CLASSIFICATION_COLUMN = 382

def remove_header_and_normalize_train(data):
    """
    Strips header, classifcation, first N elements, and preforms a StandardScaler normalization
    on the data. returns an np array
    """
    data = np.delete(data, CLASSIFICATION_COLUMN, axis=1) # get ride of classification column 
    data = np.delete(data, 0, axis=0) # get ride of header
    data = np.delete(data, 0, axis=1) # get ride of id column
    data = np.delete(data, np.s_[0:N], axis=0)
    scaler = preprocessing.StandardScaler() #sklean library which does preprocessing. 
                                            # there are quite a few. We could try multiple
                                            # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    data = scaler.fit_transform(data)
    return data

def grab_train_Y(data):
    """
    Strips header,first N elements, and preforms and grabs the classifcation for training set.
    """
    data = np.delete(data, 0, axis=0) # get ride of header
    data = np.delete(data, np.s_[0:N], axis=0) # delete first N rows
    Y = data[:, CLASSIFICATION_COLUMN]
    return Y

def remove_header_and_normalize_ver(data):
    """
    Strips header, classifcation, Grabs only the first N elements, and preforms a StandardScaler normalization
    on the data. returns an np array
    """
    data = np.delete(data, CLASSIFICATION_COLUMN, axis=1) # get ride of classification column 
    data = np.delete(data, 0, axis=0) # get ride of header
    data = np.delete(data, 0, axis=1) # get ride of id column
    data = data[np.s_[0:N], :]
    scaler = preprocessing.StandardScaler() #sklean library which does preprocessing. 
                                            # there are quite a few. We could try multiple
                                            # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    data = scaler.fit_transform(data)
    return data

def grab_ver_Y(data):
    """
    Strips header, Grabs only first N elements, and preforms and grabs the classifcation for training set.
    """
    data = np.delete(data, 0, axis=0) # get ride of header
    data =  data[np.s_[0:N], :] # take first N rows
    Y = data[:, CLASSIFICATION_COLUMN]
    return Y

def remove_header_and_normalize_test(data):
    """
    Strips Header, id column, and normalizes data with StandardScaler
    """
    data = np.delete(data, 0, axis=0) # get ride of header
    data = np.delete(data, 0, axis=1) # get ride of id column
    scaler = preprocessing.StandardScaler() #sklean library which does preprocessing. 
                                            # there are quite a few. We could try multiple
                                            # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    data = scaler.fit_transform(data)
    return data