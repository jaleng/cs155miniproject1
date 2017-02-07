import os.path
import sys
import getopt
import numpy as np
from pkl_help import get_pkl
from pkl_help import read_make_pkl
import preprocess_help as ph
from submission import make_submission_2008
from submission import make_submission_2012
from sklearn.neural_network import MLPClassifier

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")

def mlp_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds

# Pretty Slow... 
def optimize_parameters():
    max_x = -1
    max_y = -1
    score = 0.0
    train_score = []
    ver_score = []
    for x in np.arange(1, 500):
        train_score.append([])
        ver_score.append([])
        for y in np.arange(1, x+1):
            model = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(x,y), random_state=1,
                     shuffle=True, warm_start=1)
            model.fit(X_train_2008, Y_train_2008)
            train_score[x-1].append(model.score(X_train_2008, Y_train_2008))
            ver_score[x-1].append(model.score(X_ver, Y_ver))
            if ver_score[x-1][-1] > score:
                score = ver_score[x-1][-1]
                max_x = x
                max_y = y
                print "X: ", x, ". Y: ", y, ":"
                print "Training Score: ", train_score[x-1][-1], ". Verification Score: ", \
                     ver_score[x-1][-1]
    
#Example params given. Will be decided by optimizing.
def gen_mlp():
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(7,3), random_state=1, shuffle=True, warm_start=1, verbose=1)
    clf.fit(X_train_2008, Y_train_2008)
    return clf

# Save Model 
mlp = read_make_pkl("saved_objs/mlp.pkl", gen_mlp)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'mlpclassifier.py [-o [2008] [2012] [tune]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'mlpclassifer.py [-o [2008] [2012] [tune]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                # DEBUG
                print("mlp.predict(X_test_2008).shape" +
                      str(mlp.predict(X_test_2008).shape))
                make_submission_2008("submissions/mlp_2008.csv", 
                                      mlp_modified_predict(mlp, X_test_2008))
            elif (arg == "2012"):
                make_submission_2012("submissions/adaboost_2012.csv", 
                                      mlp_modified_predict(mlp, X_test_2012))
            elif (arg == "tune"):
                optimize_parameters()
if __name__ == "__main__":
    main(sys.argv[1:])