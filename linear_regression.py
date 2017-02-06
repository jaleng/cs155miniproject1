# linear_regress.py
import numpy as np
from pkl_help import get_pkl
from pkl_help import read_make_pkl
from sklearn.linear_model import LinearRegression
import sys
import getopt
from submission import make_submission_2008
from submission import make_submission_2012
from round_predictions import round_predictions

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")


# Write function to generate model
def gen_lin_reg_model():
    model = LinearRegression()
    model.fit(X_train_2008, Y_train_2008)
    # DEBUG
    print("d0: " + str(Y_train_2008.shape))
    print("d1: " + str(model.predict(X_test_2008).shape))
    print("d2: " + str(model.coef_.shape))
    #
    return model

def lin_reg_modified_predict(model, X):
    preds = model.predict(X).reshape(-1, 1)
    mpreds = round_predictions(preds)
    # Debug
    print("mpreds.shape: " + str(mpreds.shape))
    return mpreds

# Save model
lin_reg_model = read_make_pkl("saved_objs/lin_reg.pkl", gen_lin_reg_model)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'linear_regression.py [-o [2008] [2012]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'linear_regression.py [-o [2008] [2012]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                # DEBUG
                print("lin_reg_model.predict(X_test_2008).shape" +
                      str(lin_reg_model.predict(X_test_2008).shape))
                make_submission_2008("submissions/linear_regression_2008.csv",
                                     lin_reg_modified_predict(lin_reg_model, X_test_2008))
            elif (arg == "2012"):
                make_submission_2012("submissions/linear_regression_2012.csv",
                                     lin_reg_modified_predict(lin_reg_model, X_test_2012))
if __name__ == "__main__":
    main(sys.argv[1:])
