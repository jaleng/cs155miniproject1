import os.path
import numpy as np
import pkl_help
from pkl_help import read_make_pkl
from sklearn import svm
import preprocess_help as ph
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

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


params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']

X = X_train_2008
Y = np.ravel(Y_train_2008)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

for s in scores:
    print("Tuning paramets for " + str(s) + "score\n:")
    clf = GridSearchCV(SVC(C=1), params, cv=5, scoring=s)
    clf.fit(X_train, y_train)

    print ("Best params: \n")
    print(clf.best_params_)
    print("Grid Score: ")
    mean = clf.cv_results_['mean_test_score']
    std = clf.cv_results_['std_test_score']
    for m, s, p in zip(mean, std, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (m, s * 2, p))
    print("Sklearn Report: When trained on entire training set")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))