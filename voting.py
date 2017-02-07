import os.path
import sys
import getopt
import numpy as np
from pkl_help import get_pkl
from pkl_help import read_make_pkl
import preprocess_help as ph
from sklearn.ensemble import VotingClassifier
from submission import make_submission_2008
from submission import make_submission_2012

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")

# Grab all models we have.
ridge = get_pkl("saved_objs/ridge.pkl")
lasso = get_pkl("saved_objs/lasso.pkl")



def grad_boosting_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds

# Pretty Slow... 
def optimize_parameters():
    max_depth = -1
    max_est = -1
    score = 0.0
    train_score = []
    ver_score = []
    for estimators in np.arange(1, 500):
        train_score.append([])
        ver_score.append([])
        for depth in np.arange(1, 20):
            model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                         n_estimators=estimators)
            model.fit(X_train_2008, Y_train_2008)
            train_score[estimators-1].append(model.score(X_train_2008, Y_train_2008))
            ver_score[estimators-1].append(model.score(X_ver, Y_ver))
            if ver_score[estimators-1][-1] > score:
                score = ver_score[estimators-1][-1]
                max_depth = depth
                max_est = estimators
                print "Estimators: ", estimators, ". Max Depth: ", depth, ":"
                print "Training Score: ", train_score[estimators-1][-1], ". Verification Score: ", \
                     ver_score[estimators-1][-1]
    return (max_depth, max_est)
    
#Example params given. Will be decided by optimizing.
def gen_grad_boost():
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(X_train_2008, Y_train_2008)
    return clf

# Save Model 
grad_boost = read_make_pkl("saved_objs/adaboost.pkl", gen_adaboost)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'gradientboosting.py [-o [2008] [2012] [tune]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'gradientboosting.py [-o [2008] [2012] [tune]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                # DEBUG
                print("grad_boost.predict(X_test_2008).shape" +
                      str(grad_boost.predict(X_test_2008).shape))
                make_submission_2008("submissions/grad_boost_2008.csv", 
                                      adaboost_modified_predict(grad_boost, X_test_2008))
            elif (arg == "2012"):
                make_submission_2012("submissions/grad_boost_2012.csv", 
                                      adaboost_modified_predict(grad_boost, X_test_2012))
            elif (arg == "tune"):
                optimize_parameters()
if __name__ == "__main__":
    main(sys.argv[1:])