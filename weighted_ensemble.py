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
from round_predictions import round_predictions

X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/Y_train_2008.pkl")
X_test_2008 = get_pkl("saved_objs/X_test_2008.pkl")
X_test_2012 = get_pkl("saved_objs/X_test_2012.pkl")
X_ver = get_pkl("saved_objs/X_ver_2008.pkl")
Y_ver = get_pkl("saved_objs/Y_ver_2008.pkl")


def modified_predict(model, Y):
    preds = model.predict(Y).reshape(-1, 1)
    # Debug
    print("preds.shape: " + str(preds.shape))
    return preds

xgb_score = 0.77900
adaboost_ran_forest_score = 0.77838
adaboost_score = 0.77525
lasso_score = 0.77225
mlp_score = 0.77200
rand_forest_score = 0.76688
ridge_score = 0.77325

lasso = get_pkl("saved_objs/lasso.pkl")
ridge = get_pkl("saved_objs/ridge.pkl")
xgb_2008 = get_pkl("saved_objs/xgb_100k_estimators_preds_2008.pkl")
xgb_2012 = get_pkl("saved_objs/xgb_100k_estimators_preds_2012.pkl")
adaboost = get_pkl("saved_objs/adaboost.pkl")
adaboost_ran_forest = get_pkl("saved_objs/adaboost_ran_forest.pkl")
mlp = get_pkl("saved_objs/mlp.pkl")
rand_forest = get_pkl("saved_objs/rand_forest.pkl")

def pred_2008():
    lasso_unrounded = modified_predict(lasso, X_test_2008)
    ridge_unrounded = modified_predict(ridge, X_test_2008)
    mlp_unrounded = modified_predict(mlp, X_test_2008)
    xgb_unrouned = xgb_2008.reshape(-1, 1)
    rand_forest_unrounded = modified_predict(rand_forest, X_test_2008)
    adaboost_ran_forest_unrounded = modified_predict(adaboost_ran_forest, X_test_2008)
    adaboost_unrounded = modified_predict(adaboost, X_test_2008)

    total = xgb_score + adaboost_ran_forest_score + adaboost_score + mlp_score + \
        lasso_score + rand_forest_score + ridge_score
    
    lasso_weight = lasso_score / total
    ridge_weight = ridge_score / total
    xgb_weight = xgb_score / total
    adaboost_ran_forest_weight = adaboost_ran_forest_score / total
    adaboost_weight = adaboost_score / total
    mlp_weight = mlp_score / total
    rand_forest_weight = rand_forest_score / total

    lasso_2008 = lasso_weight * lasso_unrounded
    ridge_2008 = ridge_weight * ridge_unrounded
    xgb_2008_weighted = xgb_weight * xgb_unrouned
    adaboost_ran_forest_2008 = adaboost_ran_forest_weight * adaboost_ran_forest_unrounded
    adaboost_2008 = adaboost_weight * adaboost_unrounded
    mlp_2008 = mlp_weight * mlp_unrounded
    rand_forest_2008 = rand_forest_weight * rand_forest_unrounded

    lasso_unrounded = 0
    ridge_unrounded = 0
    mlp_unrounded = 0
    xgb_unrouned = 0
    rand_forest_unrounded = 0
    adaboost_ran_forest_unrounded = 0
    adaboost_unrounded = 0

    print "Starting Adding"
    temp1 = np.add(np.add(lasso_2008, ridge_2008), xgb_2008_weighted)
    print "Halway Through Adding"
    temp2 =np.add(np.add(np.add(adaboost_ran_forest_2008, adaboost_2008), mlp_2008), rand_forest_2008)
    ensemble = np.add(temp1, temp2)
    print "Done Adding!"

    temp1 = 0
    temp2 = 0

    print ensemble
    ensemble = round_predictions(ensemble)
    print "Min of ensemble: ", np.min(ensemble), ". Max: ", np.max(ensemble)
    return ensemble

def pred_2012():
    lasso_unrounded = modified_predict(lasso, X_test_2012)
    ridge_unrounded = modified_predict(ridge, X_test_2012)
    mlp_unrounded = modified_predict(mlp, X_test_2012)
    xgb_unrouned = xgb_2012.reshape(-1, 1)
    rand_forest_unrounded = modified_predict(rand_forest, X_test_2012)
    adaboost_ran_forest_unrounded = modified_predict(adaboost_ran_forest, X_test_2012)
    adaboost_unrounded = modified_predict(adaboost, X_test_2012)

    total = xgb_score + adaboost_ran_forest_score + adaboost_score + mlp_score + \
        lasso_score + rand_forest_score + ridge_score
    
    lasso_weight = lasso_score / total
    ridge_weight = ridge_score / total
    xgb_weight = xgb_score / total
    adaboost_ran_forest_weight = adaboost_ran_forest_score / total
    adaboost_weight = adaboost_score / total
    mlp_weight = mlp_score / total
    rand_forest_weight = rand_forest_score / total

    lasso_2012 = lasso_weight * lasso_unrounded
    ridge_2012 = ridge_weight * ridge_unrounded
    xgb_2012_weighted = xgb_weight * xgb_unrouned
    adaboost_ran_forest_2012 = adaboost_ran_forest_weight * adaboost_ran_forest_unrounded
    adaboost_2012 = adaboost_weight * adaboost_unrounded
    mlp_2012 = mlp_weight * mlp_unrounded
    rand_forest_2012 = rand_forest_weight * rand_forest_unrounded

    lasso_unrounded = 0
    ridge_unrounded = 0
    mlp_unrounded = 0
    xgb_unrouned = 0
    rand_forest_unrounded = 0
    adaboost_ran_forest_unrounded = 0
    adaboost_unrounded = 0

    print "Starting Adding"
    temp1 = np.add(np.add(lasso_2012, ridge_2012), xgb_2012_weighted)
    print "Halway Through Adding"
    temp2 =np.add(np.add(np.add(adaboost_ran_forest_2012, adaboost_2012), mlp_2012), rand_forest_2012)
    ensemble = np.add(temp1, temp2)
    print "Done Adding!"

    temp1 = 0
    temp2 = 0

    print ensemble
    ensemble = round_predictions(ensemble)
    print "Min of ensemble: ", np.min(ensemble), ". Max: ", np.max(ensemble)
    return ensemble

make_submission_2008("submissions/ensemble_2008.csv", pred_2008())
make_submission_2012("submissions/ensemble_2012.csv", pred_2012())