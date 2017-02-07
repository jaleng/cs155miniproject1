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
from sklearn.linear_model import RidgeClassifierCV

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")

# Grab all models we have.
ridge = RidgeClassifierCV().fit(X_train_2008, Y_train_2008)
# lasso = get_pkl("saved_objs/lasso.pkl")
mlp = get_pkl("saved_objs/mlp.pkl")
rand_forest = get_pkl("saved_objs/rand_forest.pkl")
adaboost = get_pkl("saved_objs/adaboost.pkl")
# knn = get_pkl("saved_objs/knearest.pkl")


def voting_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds
    
def gen_voting():
    print "Making Voting Classifier"
    clf =VotingClassifier(estimators=[('ridge', ridge), 
                                      ('mlp', mlp),
                                      ('rand_forest', rand_forest),
                                      ('adaboost', adaboost)],
                                       voting='hard')
    print "Training Voting Classifier"
    clf.fit(X_train_2008, Y_train_2008)
    print "Scoring Voting Classifier"
    print "Training Score: ", clf.score(X_train_2008, Y_train_2008)
    print "Ver Score: ", clf.score(X_ver, Y_ver)    
    return clf

# Save Model 
voting = read_make_pkl("saved_objs/voting.pkl", gen_voting)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'voting.py [-o [2008] [2012]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'voting.py [-o [2008] [2012]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                # DEBUG
                print("voting.predict(X_test_2008).shape" +
                      str(voting.predict(X_test_2008).shape))
                make_submission_2008("submissions/voting_2008.csv", 
                                      voting_modified_predict(voting, X_test_2008))
            elif (arg == "2012"):
                make_submission_2012("submissions/voting_2012.csv", 
                                      voting_modified_predict(voting, X_test_2012))
if __name__ == "__main__":
    main(sys.argv[1:])