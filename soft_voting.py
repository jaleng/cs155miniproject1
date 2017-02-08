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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")
X_ver_2008 = X_ver
Y_ver_2008 = Y_ver


def ada_random_forest():
    return AdaBoostClassifier(RandomForestClassifier(max_depth=1), n_estimators=500, learning_rate=1).fit(X_train_2008, Y_train_2008)

# Grab all models we have.
lasso = get_pkl("saved_objs/lasso.pkl")
ridge = get_pkl("saved_objs/ridge.pkl")
# mlp = get_pkl("saved_objs/mlp.pkl")
# rand_forest = get_pkl("saved_objs/rand_forest.pkl")
adaboost = get_pkl("saved_objs/adaboost.pkl")

adaboost_ran_forest = read_make_pkl("saved_objs/adaboost_ran_forest.pkl", ada_random_forest)

print "Adaboost Random Forest Training Error: ", adaboost_ran_forest.score(X_train_2008, Y_train_2008)
print "Adaboost Random Forest Val Error: ", adaboost_ran_forest.score(X_ver, Y_ver)

# def gen_xgb():
#     evals = [(X_ver_2008, Y_ver_2008)]
#     model = xgb.XGBRegressor(n_estimators=1000)
#     model.fit(X_train_2008, Y_train_2008, eval_set=evals,
#               early_stopping_rounds=20, verbose=True)
#     return model

# xgb_model = read_make_pkl("saved_objs/xgb_1000_estimators.pkl", gen_xgb)

# knn = get_pkl("saved_objs/knearest.pkl")


def voting_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds
    
def gen_voting():
    print "Making Voting Classifier"
    clf =VotingClassifier(estimators=[('ridge', ridge), 
                                      ('adaboost', adaboost),
                                      ('adaboost_ran_forest', adaboost_ran_forest),
                                      ('lasso', adaboost_ran_forest),                                      
                                      ],
                                       voting='soft')
    print "Training Voting Classifier"
    clf.fit(X_train_2008, Y_train_2008)
    print "Scoring Voting Classifier"
    print "Training Score: ", clf.score(X_train_2008, Y_train_2008)
    print "Ver Score: ", clf.score(X_ver, Y_ver)    
    return clf

# Save Model 
voting = read_make_pkl("saved_objs/soft_voting.pkl", gen_voting)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'voting_classification.py [-o [2008] [2012]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'voting_classification.py [-o [2008] [2012]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                # DEBUG
                print("soft_voting.predict(X_test_2008).shape" +
                      str(voting.predict(X_test_2008).shape))
                make_submission_2008("submissions/soft_voting_2008.csv", 
                                      voting_modified_predict(voting, X_test_2008))
            elif (arg == "2012"):
                make_submission_2012("submissions/soft_voting_2012.csv", 
                                      voting_modified_predict(voting, X_test_2012))
if __name__ == "__main__":
    main(sys.argv[1:])