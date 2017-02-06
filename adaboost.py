import os.path
import sys
import getopt
import numpy as np
from pkl_help import get_pkl
from pkl_help import read_make_pkl
import preprocess_help as ph
from sklearn.ensemble import AdaBoostClassifier
from submission import make_submission_2008
from submission import make_submission_2012
from sklearn.tree import DecisionTreeClassifier

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")

def adaboost_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds

# Tried a few different parameters - should optimize further.
def optimize_parameters():
    max_depth = -1
    max_est = -1
    train_score = []
    ver_score = []
    for estimators in np.arange(1, 500):
        train_score.apppend([])
        ver_score.apppend([])
        for depth in np.arange(1, 20):
            adaboost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                         n_estimators=estimators)
            adaboost_model.fit(X_train_2008, Y_train_2008)
            train_score[estimators-1].apppend(adaboost_model.score(X_train_2008, Y_train_2008))
            ver_score[estimators-1].apppend(adaboost_model.score(X_ver, Y_ver))
            print "Estimators: ", estimators, ". Max Depth: ", depth, ":"
            print "Training Score: ", train_score[estimators-1][-1], ". Verification Score: ", ver_score[estimators-1][-1]

optimize_parameters()

# print "Training Score: ", adaboost_model.score(X, Y)
# print "Verification Score: ", adaboost_model.score(X_ver, Y_ver)

# make_submission_2008("submissions/adaboost_2008.csv", adaboost_modified_predict(adaboost_model, X_test_2008))