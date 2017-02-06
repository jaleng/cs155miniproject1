import os.path
import sys
import getopt
import numpy as np
import pkl_help
from pkl_help import get_pkl
from pkl_help import read_make_pkl
import preprocess_help as ph
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from submission import make_submission_2008
from submission import make_submission_2012

# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")




def tune_and_graph():
    depth_in = []
    depth_out = []
    sample_leaf_in = []
    sample_leaf_out = []
    n_estimators_in = []
    n_estimators_out = []

    ############################  Tuning Parameters ####################################
    for n in np.arange(1, 20):
        clf = RandomForestClassifier(max_depth=n, n_estimators=10, max_features='auto')
        clf.fit(X_train_2008, Y_train_2008)
        depth_in.append(clf.score(X_train_2008, Y_train_2008))
        depth_out.append(clf.score(X_ver, Y_ver))
        print "Ran_Forest Max Depth: ", n, ": Training Score: " , depth_in[-1], \
            ": Test Score: " , depth_out[-1]

    print "################################################################################\n\n"

    for n in np.arange(1, 1000, 20):
        clf = RandomForestClassifier(min_samples_leaf=n, n_estimators=10, max_features='auto')
        clf.fit(X_train_2008, Y_train_2008)
        sample_leaf_in.append(clf.score(X_train_2008, Y_train_2008))
        sample_leaf_out.append(clf.score(X_ver, Y_ver))
        print "Ran_Forest Min Leaf: ", n, ": Training Score: " , sample_leaf_in[-1], ": Test Score: " \
          , sample_leaf_out[-1]

    print "################################################################################\n\n"
    start = 30
    for x in np.arange(start, 50):
        n_estimators_in.append([])
        n_estimators_out.append([])
      
        for n in np.arange(40, 50):
            clf = RandomForestClassifier(max_depth=n, n_estimators=x, max_features='auto')
            clf.fit(X_train_2008, Y_train_2008)
            n_estimators_in[x-start].append(clf.score(X_train_2008, Y_train_2008))
            n_estimators_out[x-start].append(clf.score(X_ver, Y_ver))
            print "Ran_Forest ", x, "Ests: Max Depth: ", n ,"\nTraining Score: " , \
                n_estimators_in[x-start][-1], ": Test Score: " , n_estimators_out[x-start][-1]
        print "################################################################################\n\n"
    #############################################################################################
    ################################ Graphs for Report #########################################
    ############################################################################################
    plt.title("Ein vs Eout as Min Sample Leaf Increases")
    plt.plot(np.arange(1, 1000, 20), sample_leaf_in, color="b", label="Training Set Score")
    plt.plot(np.arange(1, 1000, 20), sample_leaf_out, color="r", label="Validation Set Score")
    plt.ylabel("Score - Accuracy")
    plt.xlabel("Minimum Sample Leaf Size")
    plt.legend()
    plt.savefig("min_sample_leaf")
    plt.show()
    plt.close()

    plt.title("Ein vs Eout as Max Depth Increases")
    plt.plot(np.arange(1, 20), depth_in, color="b", label="Training Set Error")
    plt.plot(np.arange(1, 20), depth_out, color="r", label="Validation Set Error")
    plt.ylabel("Score - Accuracy")
    plt.xlabel("Max Depth Error")
    plt.legend()
    plt.savefig("max_depth")
    plt.show()
    plt.close() 

    ###########################################################################################

def gen_rand_forest():
    model = RandomForestClassifier(max_depth=15, n_estimators=30, max_features='auto')
    model.fit(X_train_2008, Y_train_2008)
    return model


def rand_forest_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds

# Save Model
rand_forest_model = read_make_pkl("saved_objs/rand_forest.pkl", gen_rand_forest)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ho:",["output="])
    except getopt.GetoptError:
        print 'random_forest.py [-o [2008] [2012] [graphs]]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'random_forest.py [-o [2008] [2012] [graphs]]'
            sys.exit()
        elif opt in ("-o", "--output"):
            if (arg == "2008"):
                # DEBUG
                print("rand_forest_model.predict(X_test_2008).shape" +
                      str(rand_forest_model.predict(X_test_2008).shape))
                make_submission_2008("submissions/random_forest_2008.csv", 
                                      rand_forest_modified_predict(rand_forest_model, X_test_2008))
            elif (arg == "2012"):
                make_submission_2012("submissions/random_forest_2012.csv", 
                                      rand_forest_modified_predict(rand_forest_model, X_test_2012))
            elif (arg == "graphs"):
                tune_and_graph()
if __name__ == "__main__":
    main(sys.argv[1:])
