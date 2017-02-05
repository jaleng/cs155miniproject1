# linear_regress.py
import numpy as np
from pkl_help import get_pkl
from pkl_help import read_make_pkl
from sklearn.linear_model import LinearRegression
# Get training data
X_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")
Y_train_2008 = get_pkl("saved_objs/X_train_2008.pkl")

# Write function to generate model
def gen_lin_reg_model():
    model = LinearRegression()
    model.fit(X_train_2008, Y_train_2008)
    return model

# Save model
lin_reg_model = read_make_pkl("saved_objs/lin_reg.pkl", gen_lin_reg_model)

# Write function to output predictions on 2008 test data

# Write function to output predictions on 2012 test data

