import os.path
import numpy as np
import pkl_help
from pkl_help import read_make_pkl
from sklearn.svm import SVC
import preprocess_help as ph
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from submission import make_submission_2008
from sklearn.tree import DecisionTreeClassifier


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
X_test_2008 = read_make_pkl("saved_objs/X_test_2008.pkl",
                           lambda: ph.remove_header_and_normalize_test(test_2008),
                           compress=True)


X = X_train_2008
Y = np.ravel(Y_train_2008)

X_ver = X_ver_2008
Y_ver = np.ravel(Y_ver_2008)

def adaboost_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds

# Tried a few different parameters - should optimize further.

adaboost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         n_estimators=200)
adaboost_model.fit(X, Y)

print "Training Score: ", adaboost_model.score(X, Y)
print "Verification Score: ", adaboost_model.score(X_ver, Y_ver)


make_submission_2008("submissions/adaboost_2008.csv", adaboost_modified_predict(adaboost_model, X_test_2008))