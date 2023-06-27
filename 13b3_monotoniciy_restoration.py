import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import optuna
import multiprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
import random
import TFM_functions as tfm
from scipy.optimize import curve_fit

#%% IMPORT DATA

p = pd.read_csv(r'Data\p.csv')
p = p['x'].tolist()

folder = "Data/Preprocessed_inputs/"
sufix=''
X_train = pd.read_csv(folder + 'X_train' + sufix + '.csv')
Y_train = pd.read_csv(folder + 'Y_train' + sufix + '.csv')
X_test = pd.read_csv(folder + 'X_test' + sufix + '.csv')
Y_test = pd.read_csv(folder + 'Y_test' + sufix + '.csv')

X_train['datetime'] = pd.to_datetime(X_train['datetime'])
X_train.set_index('datetime', inplace=True)
X_test['datetime'] = pd.to_datetime(X_test['datetime'])
X_test.set_index('datetime', inplace=True)
Y_train['datetime'] = pd.to_datetime(Y_train['datetime'])
Y_train.set_index('datetime', inplace=True)
Y_test['datetime'] = pd.to_datetime(Y_test['datetime'])
Y_test.set_index('datetime', inplace=True)

# Y_train_preds
Y_train_pred = np.loadtxt("Data/13_monotonicity_restoration/13_Y_train_preds.csv", delimiter=",")
Y_train_pred = pd.DataFrame(Y_train_pred)
Y_train_pred.index = Y_train.index
Y_train_pred.columns = Y_train.columns

# Y_test_preds
Y_test_pred = np.loadtxt("Data/Preds/10_HGB_preds.csv", delimiter=",")
Y_test_pred = pd.DataFrame(Y_test_pred)
Y_test_pred.index = Y_test.index
Y_test_pred.columns = Y_test.columns

del folder, sufix
del X_train, X_test
 
    
#%%

Y_test_pred_corrected = tfm.restore_monotonicity(Y_test_pred, max_iter=20)


#%% MEASURE ERROR

# Measure errors
stats, e_mae1, e_rmse1 = tfm.errors_analysis(Y_test_pred, Y_test, totals=True)
stats, e_mae2, e_rmse2 = tfm.errors_analysis(Y_test_pred_corrected, Y_test, totals=True)



