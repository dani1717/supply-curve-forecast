#%% IMPORTS --------------------------------------------------------------------
import pandas as pd
import numpy as np
import csv
import random
import datetime
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import myFunctions_Dani as dani
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import time
import optuna
import multiprocessing
import pickle
import matplotlib.dates as mdates
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import TFM_functions as tfm

#%% LOAD DATA --------------------------------------------------------------------

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

# Original data
file_name = "Data/Other/04_Preprocessing_data_original.pickle"
with open(file_name, "rb") as f:
    data_original = pickle.load(f)
  
del folder, file_name, f

#%% NORMALIZE DATA

Y_test_unscaled = Y_test.copy()

# Scale Y  NO ES NECESARIOOOOOOOOOOOOOOOOO
scaler = StandardScaler()
Y_train = scaler.fit_transform(Y_train)
Y_test = scaler.transform(Y_test)

#Convert to dataframe again
Y_train = pd.DataFrame(Y_train)
Y_train.index = X_train.index
Y_train.columns = Y_test_unscaled.columns
Y_test = pd.DataFrame(Y_test)
Y_test.index = X_test.index
Y_test.columns = Y_test_unscaled.columns

# Scale X
scaler2 = StandardScaler()
cols_to_scale = ['mDiario_Q', 'mDiario_price', 'r1', 'r2', 'w1', 'w2', 'w3', 'w4', 'w5',
       'w6', 'w7', 'w8', 'w9', 'w10', 'gas_price', 'day_of_month',
       'lag24_pc1', 'lag24_pc2', 'lag24_pc3', 'lag24_pc4', 'lag24_pc5',
       '2weeksEvol_1', '2weeksEvol_2', '2weeksEvol_3', '2weeksEvol_4']
X_train[cols_to_scale] = scaler2.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler2.transform(X_test[cols_to_scale])

file_name = "E:/Temporal/10_HistGB_scaler.pickle"
with open(file_name, "wb") as f:  
    pickle.dump(scaler, f)


#%% HPO

n_cores = multiprocessing.cpu_count()
tscv = TimeSeriesSplit(n_splits=2)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.1, 1)
    max_iter = trial.suggest_int("max_iter", 50, 200)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 60)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 80)

    model = HistGradientBoostingRegressor(random_state=100482712,learning_rate=learning_rate,
                                          max_iter=max_iter,max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    multioutput_model = MultiOutputRegressor(model)

    error_avg = np.sqrt(-np.mean(cross_val_score(multioutput_model,X_train.values,Y_train.values,
                cv=tscv,scoring="neg_mean_squared_error",error_score="raise",n_jobs=n_cores)))

    return error_avg

budget = 30
study = optuna.create_study(direction = "minimize")   # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=2)
study.optimize(objective, n_trials = budget)
print("The best set of parameters is: ", study.best_params)
print("The rmse score computed with inner-evaluation is:", study.best_value)
print(f"The optimal value is reached in {study.best_trial.number} iterations")

#%% PREDICTION ---------------------------------------

#Trial 21 finished with value: 0.6671311307229449 and parameters: {'learning_rate': 0.1043940232366647, 'max_iter': 122, 'max_leaf_nodes': 10, 'max_depth': 16, 'min_samples_leaf': 80}
learning_rate = 0.1043940232366647
max_iter = 122
max_leaf_nodes = 10
max_depth = 16
min_samples_leaf = 80

model_HGB = HistGradientBoostingRegressor(random_state=100482712,learning_rate=learning_rate,
                                      max_iter=max_iter,max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
model = MultiOutputRegressor(model_HGB)
model.fit(X_train, Y_train)

file_name = "E:/Temporal/10_HistGB_model.pickle"
with open(file_name, "wb") as f:  # Save model in HDD 
    #pickle.dump(model, f)
with open(file_name, "rb") as f:  # Read model from HDD
    model = pickle.load(f)

# Predict
Y_pred = model.predict(X_test)
Y_pred = pd.DataFrame(Y_pred)
Y_pred.index = Y_test.index
Y_pred.columns = Y_test.columns

# Unscale and save
Y_pred_unscaled = pd.DataFrame(scaler.inverse_transform(Y_pred))
Y_pred_unscaled.columns = Y_test.columns
Y_pred_unscaled.index = Y_test.index
#np.savetxt('Data/Preds/10_HGB_preds.csv', Y_pred_unscaled, delimiter=',')

#%% COMPARISON AND METRICS

# Compare series
tfm.plotPreds(Y_test_unscaled, Y_pred_unscaled, n_days=4, plotLag24=True) 
# Compare curves
tfm.compareCurve(Y_test_unscaled, Y_pred_unscaled,plotLag24=True)

stats, e_mae, e_rmse = tfm.errors_analysis(Y_test_unscaled, Y_pred_unscaled, totals=True)


# 2019
Y_pred2 = Y_pred_unscaled.iloc[0:365*24]
Y_test2 = Y_test_unscaled.iloc[0:365*24]
stats, e_mae, e_rmse = tfm.errors_analysis(Y_test2, Y_pred2, totals=True)

tfm.compareCurve(Y_test2, Y_pred2,plotLag24=True)












