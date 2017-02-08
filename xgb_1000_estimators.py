import xgboost as xgb
import numpy as np
from sklearn.linear_model import RidgeCV
from pkl_help import get_pkl
from pkl_help import read_make_pkl
from round_predictions import round_predictions
import sys
import getopt
from submission import make_submission_2008
from submission import make_submission_2012

# Get data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_ver_2008 = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver_2008 = get_pkl("saved_objs/Y_ver_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")

#dtrain = xgb.DMatrix(X_train_2008, label=Y_train_2008)
#dver = xgb.DMatrix(X_ver_2008, label=Y_ver_2008)
#bst = xgb.train(plst, dtrain, evals=evallist)

# Generate xgb model
def gen_xgb():
    evals = [(X_ver_2008, Y_ver_2008)]
    model = xgb.XGBRegressor(n_estimators=1000)
    model.fit(X_train_2008, Y_train_2008, eval_set=evals,
              early_stopping_rounds=20, verbose=True)
    return model

xgb_model = read_make_pkl("saved_objs/xgb_1000_estimators.pkl", gen_xgb)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'xgb_1000_estimators.py [-o [2008] [2012]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'xgb_1000_estimators.py [-o [2008] [2012]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                preds = round_predictions(xgb_model.predict(X_test_2008))
                # DEBUG
                print("preds.shape" +
                      str(preds.shape))
                make_submission_2008("submissions/xgb_1000_estimators_2008.csv",
                                     preds)
            elif (arg == "2012"):
                preds = round_predictions(xgb_model.predict(X_test_2012))
                make_submission_2012("submissions/xgb_1000_estimators_2012.csv",
                                     preds)
if __name__ == "__main__":
    main(sys.argv[1:])
    preds = round_predictions(xgb_model.predict(X_train_2008))
    train_error = np.mean(preds == Y_train_2008)
    print("XGB_1000_estimators train error = " + str(train_error))

    ver_preds = round_predictions(xgb_model.predict(X_ver_2008))
    ver_error = np.mean(ver_preds == Y_ver_2008)
    print("XGB_1000_estimators ver error = " + str(ver_error))
