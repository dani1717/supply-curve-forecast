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


#%% HPO

n_cores = multiprocessing.cpu_count()
tscv = TimeSeriesSplit(n_splits=2)

def objective(trial):
    
    n_estimators = trial.suggest_int("n_estimators", 350,1000)
    max_depth = trial.suggest_int("max_depth", 15,55)
    min_samples_split = trial.suggest_int("min_samples_split",4,18)
    
    
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



#%% PREDICTION ---------------------------------------

n_estimators = 617
max_depth = 39
min_samples_split = 10
n_cores = multiprocessing.cpu_count()


model = RandomForestRegressor(random_state = 100482712,
                                 n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split,
                                 n_jobs = n_cores)
model.fit(X_train, Y_train)
file_name = "E:/Temporal/07_RF_RFmodel.pickle"
with open(file_name, "wb") as f:  # Save model in HDD 
    #pickle.dump(model, f)
with open(file_name, "rb") as f:  # Read model from HDD
    model = pickle.load(f)

# Importance of features
n_show= 25
importances = model.feature_importances_
feature_names = X_train.columns
sorted_idx = importances.argsort()[::-1]
plt.bar(range(n_show), importances[sorted_idx][:n_show])
plt.xticks(range(n_show), feature_names[sorted_idx][:n_show], rotation=90)
plt.title("Importance of features - Random Forest")
plt.show()
    
#Y_pred = model.predict(X_test)
#np.savetxt('Data/Preds/07_RF_preds.csv', Y_pred, delimiter=',')
Y_pred = np.loadtxt("Data/Preds/07_RF_preds.csv", delimiter=",")
Y_pred = pd.DataFrame(Y_pred)
Y_pred.index = Y_test.index
Y_pred.columns = Y_test.columns

#%% COMPARISON AND METRICS

def plotPreds(Y_test, Y_pred, serie=None, from_date=None, to_date=None, n_days=None, plotLag24=False):
    # Plots the comparison of prediction and real series
    
    if n_days is None:
        n_days = 7
    if from_date is None:
        from_date = Y_test.index[random.randint(0, len(Y_test)-n_days*24)]  # We sample a random row from Y_test
    else:
        from_date = datetime.datetime.strptime(from_date, "%Y-%m-%d")
    if to_date is None:
            to_date = from_date + datetime.timedelta(days=n_days)
    if serie is None:
        serie = random.choice(list(Y_test.columns))
    
    Y_pred = pd.DataFrame(Y_pred)
    real_values = Y_test.loc[from_date:to_date, serie]
    rows = list(range(Y_test.index.get_loc(from_date), Y_test.index.get_loc(from_date) + len(real_values)))
    col = Y_test.columns.get_loc(serie)
    pred_values = Y_pred.iloc[rows, col]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_values, color='blue', label='Real values', linewidth=2)
    ax.plot(real_values.index, pred_values, color='red', label='Predicted values', linestyle='--')
    if plotLag24:
        lag_values = Y_test.loc[from_date-datetime.timedelta(hours=24):to_date-datetime.timedelta(hours=24), serie]
        lag_values.index = real_values.index
        ax.plot(lag_values.index, lag_values, color='green', label='lag24 estimator', linestyle='--')
    ax.set_ylabel(serie)
    ax.set_title('Prediction for ' + serie)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.legend()
    plt.show()


plotPreds(Y_test, Y_pred, n_days=4, plotLag24=True)  # We are comparing scaled data

# UNSCALE AND COMPARE
Y_pred_unscaled = pd.DataFrame(scaler.inverse_transform(Y_pred))
Y_pred_unscaled.columns = Y_test.columns
Y_pred_unscaled.index = Y_test.index

Y_test_original = data_original.loc[Y_pred_unscaled.index]
Y_test_original = Y_test_original.iloc[:,0:50]

plotPreds(Y_test_original, Y_pred_unscaled, n_days=2, plotLag24=True)

import TFM_functions as tfm
stats, mae, mse = tfm.errors_analysis(Y_test, Y_pred, totals=True)














