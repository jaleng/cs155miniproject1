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
from sklearn.neighbors import KNeighborsClassifier

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")

def knearest_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds


def optimize_parameters():
    score = 0.0
    neighbor = 0
    train_score = []
    ver_score = []
    for n in np.arange(1, 500):
        model = KNeighborsClassifier(n_neighbors=n)
        print "Training Model with ", n, "neighbors."
        model.fit(X_train_2008, Y_train_2008)
        train_score.append(model.score(X_train_2008, Y_train_2008))
        ver_score.append(model.score(X_ver, Y_ver))
        if ver_score[-1] > score:
            score = ver_score[-1]
            neighbor = n
            print "Number of neighbors: ", neighbor,
            print "Training Score: ", train_score[-1], ". Verification Score: ", \
                 ver_score[-1]
    return neighbor
    
#Example params given. Will be decided by optimizing.
def gen_knearest():
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train_2008, Y_train_2008)
    return clf

# Save Model 
knearest = read_make_pkl("saved_objs/knearest.pkl", gen_knearest)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'knearest_neighbor.py [-o [2008] [2012] [tune]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'knearest_neighbor.py [-o [2008] [2012] [tune]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                # DEBUG
                print("knearest.predict(X_test_2008).shape" +
                      str(adaboost.predict(X_test_2008).shape))
                make_submission_2008("submissions/knearest_2008.csv", 
                                      knearest_modified_predict(knearest, X_test_2008))
            elif (arg == "2012"):
                make_submission_2012("submissions/knearest_2008.csv", 
                                      knearest_modified_predict(knearest, X_test_2012))
            elif (arg == "tune"):
                optimize_parameters()
if __name__ == "__main__":
    main(sys.argv[1:])