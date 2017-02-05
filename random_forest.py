import os.path
import numpy as np
import pkl_help
from pkl_help import read_make_pkl
import preprocess_help as ph
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from submission import make_submission_2008

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
X_test_2008 = read_make_pkl("saved_objs/X_test_2008.pkl",
                           lambda: ph.remove_header_and_normalize_test(test_2008),
                           compress=True)

X = X_train_2008
Y = np.ravel(Y_train_2008)

X_ver = X_ver_2008
Y_ver = np.ravel(Y_ver_2008)

depth_in = []
depth_out = []
sample_leaf_in = []
sample_leaf_out = []
n_estimators_in = []
n_estimators_out = []


##################  Tuning Parameters ###############################################
# for n in np.arange(1, 20):
#     clf = RandomForestClassifier(max_depth=n, n_estimators=10, max_features='auto')
#     clf.fit(X, Y)
#     depth_in.append(clf.score(X, Y))
#     depth_out.append(clf.score(X_ver, Y_ver))
#     print "Ran_Forest Max Depth: ", n, ": Training Score: " , depth_in[-1], ": Test Score: " , depth_out[-1]

# print "################################################################################\n\n"

# for n in np.arange(1, 1000, 20):
#     clf = RandomForestClassifier(min_samples_leaf=n, n_estimators=10, max_features='auto')
#     clf.fit(X, Y)
#     sample_leaf_in.append(clf.score(X, Y))
#     sample_leaf_out.append(clf.score(X_ver, Y_ver))
#     print "Ran_Forest Min Leaf: ", n, ": Training Score: " , sample_leaf_in[-1], ": Test Score: " , sample_leaf_out[-1]

# print "################################################################################\n\n"
############################################################################################


def rand_forest_modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds

rand_forest_model = RandomForestClassifier(max_depth=15, n_estimators=30, max_features='auto')
rand_forest_model.fit(X, Y)

make_submission_2008("submissions/random_forest_2008.csv", rand_forest_modified_predict(rand_forest_model, X_test_2008))
# start = 30
# for x in np.arange(start, 50):
#     n_estimators_in.append([])
#     n_estimators_out.append([])
  
#     for n in np.arange(40, 50):
#         clf = RandomForestClassifier(max_depth=n, n_estimators=x, max_features='auto')
#         clf.fit(X, Y)
#         n_estimators_in[x-start].append(clf.score(X, Y))
#         n_estimators_out[x-start].append(clf.score(X_ver, Y_ver))
#         print "Ran_Forest ", x, "Ests: Max Depth: ", n ,"\nTraining Score: " , n_estimators_in[x-start][-1], ": Test Score: " , n_estimators_out[x-start][-1]
#     print "################################################################################\n\n"

#############################################################################################
################################# Graphs for Report #########################################
#############################################################################################
# plt.title("Ein vs Eout as Min Sample Leaf Increases")
# plt.plot(np.arange(1, 1000, 20), sample_leaf_in, color="b", label="Training Set Score")
# plt.plot(np.arange(1, 1000, 20), sample_leaf_out, color="r", label="Validation Set Score")
# plt.ylabel("Score - Accuracy")
# plt.xlabel("Minimum Sample Leaf Size")
# plt.legend()
# plt.savefig("min_sample_leaf")
# plt.show()
# plt.close()

# plt.title("Ein vs Eout as Max Depth Increases")
# plt.plot(np.arange(1, 20), depth_in, color="b", label="Training Set Error")
# plt.plot(np.arange(1, 20), depth_out, color="r", label="Validation Set Error")
# plt.ylabel("Score - Accuracy")
# plt.xlabel("Max Depth Error")
# plt.legend()
# plt.savefig("max_depth")
# plt.show()
# plt.close() 

############################################################################################

