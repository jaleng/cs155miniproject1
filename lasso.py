import numpy as np
from sklearn.linear_model import LassoCV
from pkl_help import get_pkl
from pkl_help import read_make_pkl
from round_predictions import round_predictions
import sys
import getopt
from submission import make_submission_2008
from submission import make_submission_2012

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")

# function to generate lasso model
def gen_lasso():
    model = LassoCV(cv=10, n_jobs=-1, verbose=True)
    model.fit(X_train_2008, Y_train_2008)
    return model

# Save model
lasso = read_make_pkl("saved_objs/lasso.pkl", gen_lasso)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'lasso.py [-o [2008] [2012]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'lasso.py [-o [2008] [2012]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                preds = round_predictions(lasso.predict(X_test_2008))
                # DEBUG
                print("lasso.predict(X_test_2008).shape" +
                      str(lasso.predict(X_test_2008).shape))
                make_submission_2008("submissions/lasso_2008.csv",
                                     preds)
            elif (arg == "2012"):
                preds = round_predictions(lasso.predict(X_test_2012))
                make_submission_2012("submissions/lasso_2012.csv",
                                     preds)
if __name__ == "__main__":
    main(sys.argv[1:])
    preds = round_predictions(lasso.predict(X_train_2008))
    train_error = np.mean(preds == Y_train_2008)
    print("Lasso train error = " + str(train_error))
