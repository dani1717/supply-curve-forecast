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
import matplotlib.dates as mdates
import pickle
import TFM_functions as tfm

#%% LOAD DATA --------------------------------------------------------------------

folder = "Data/Preprocessed_inputs/"

# Inputs
X_train = pd.read_csv(folder + 'X_train.csv')
Y_train = pd.read_csv(folder + 'Y_train.csv')
X_test = pd.read_csv(folder + 'X_test.csv')
Y_test = pd.read_csv(folder + 'Y_test.csv')

X_train['datetime'] = pd.to_datetime(X_train['datetime'])
X_train.set_index('datetime', inplace=True)
X_test['datetime'] = pd.to_datetime(X_test['datetime'])
X_test.set_index('datetime', inplace=True)
Y_train['datetime'] = pd.to_datetime(Y_train['datetime'])
Y_train.set_index('datetime', inplace=True)
Y_test['datetime'] = pd.to_datetime(Y_test['datetime'])
Y_test.set_index('datetime', inplace=True)

cols_exog = X_train.columns[0:X_train.columns.get_loc('gas_price')+1]
X_train.drop(cols_exog, axis=1, inplace=True)
X_test.drop(cols_exog, axis=1, inplace=True)

# Original data
file_name = "Data/Other/04_Preprocessing_data_original.pickle"
with open(file_name, "rb") as f:
    data_original = pickle.load(f)
    
del folder, file_name, f, cols_exog

#%% NORMALIZE DATA

Y_test_unscaled = Y_test.copy()

# Scale Y 
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


#%% HPO
n_cores = multiprocessing.cpu_count()
tscv = TimeSeriesSplit(n_splits=2)
'''
def objective(trial):
    
    n_estimators = trial.suggest_int("n_estimators", 350,900)
    max_depth = trial.suggest_int("max_depth", 30, 70)
    min_samples_split = trial.suggest_int("min_samples_split",8,30)
    
    
    model = RandomForestRegressor(random_state = 100482712, warm_start = True,
                                     n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split,
                                     n_jobs = n_cores)
    
    start_time = time.time()
    error_avg = np.sqrt(-np.mean(cross_val_score(model,X_train,Y_train,cv = tscv, scoring = "neg_mean_squared_error"
                                        , error_score = "raise", n_jobs = n_cores)))
    end_time = time.time()
    print("Execution time for trial: {:.2f} seconds".format(end_time - start_time))
    return error_avg

budget = 30
study = optuna.create_study(direction = "minimize")   # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=2)
study.optimize(objective, n_trials = budget)
print("The best set of parameters is: ", study.best_params)
print("The rmse score computed with inner-evaluation is:", study.best_value)
print(f"The optimal value is reached in {study.best_trial.number} iterations")
'''


#%% PREDICTION ---------------------------------------

# Trial 3 finished with value: 175.882547821256 and parameters: {'n_estimators': 592, 'max_depth': 44, 'min_samples_split': 27}
n_estimators = 592
max_depth = 44
min_samples_split = 27
n_cores = multiprocessing.cpu_count()


model = RandomForestRegressor(random_state = 100482712,
                                 n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split,
                                 n_jobs = n_cores)
model.fit(X_train, Y_train)

file_name = "Data/Other/06_RF_withoutExogVars_RFmodel.pickle"
with open(file_name, "wb") as f:
    pickle.dump(model, f)
with open(file_name, "rb") as f:
    model = pickle.load(f)
    
Y_pred = model.predict(X_test)
Y_pred = pd.DataFrame(Y_pred)
Y_pred.columns = Y_test.columns
Y_pred.index = Y_test.index

# Unscale and save
Y_pred_unscaled = pd.DataFrame(scaler.inverse_transform(Y_pred))
Y_pred_unscaled.columns = Y_test.columns
Y_pred_unscaled.index = Y_test.index

#%% COMPARISON AND METRICS

tfm.plotPreds(Y_test_unscaled, Y_pred_unscaled, n_days=4) 

# Statistics
stats, e_mae, e_rmse = tfm.errors_analysis(Y_test_unscaled, Y_pred_unscaled, totals=True)

X_test = X_test.iloc[0:365*24,:]
Y_test = Y_test.iloc[0:365*24,:]
