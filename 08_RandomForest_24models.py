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


# Drop  useless columns since we are creating 24 models
cols_to_drop = ['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7',
       'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13',
       'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
       'hour_20', 'hour_21', 'hour_22', 'hour_23']
X_train.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)

#%% HPO

n_cores = multiprocessing.cpu_count()
tscv = TimeSeriesSplit(n_splits=2)

def objective(trial):
    
    n_estimators = trial.suggest_int("n_estimators", 350,900)
    max_depth = trial.suggest_int("max_depth", 25, 45)
    min_samples_split = trial.suggest_int("min_samples_split",5, 18)
    
    model = RandomForestRegressor(random_state = 100482712, warm_start = True,
                                     n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split,
                                     n_jobs = n_cores)
    
    start_time = time.time()
    error_avgs = []
    n_hours = 6
    hours = random.sample(range(24), n_hours)
    hours = [1,4,7,10,13,16,19,22]
    for hour in hours:
        # We calculate the error using the model for different hours and later return the average 
        X_train_hour = X_train.loc[X_train.index.hour == hour]
        Y_train_hour = Y_train.loc[Y_train.index.hour == hour]
        error_avg = np.sqrt(-np.mean(cross_val_score(model,X_train_hour,Y_train_hour,cv = tscv, scoring = "neg_mean_squared_error"
                                            , error_score = "raise", n_jobs = n_cores)))
        error_avgs.append(error_avg)
    error_avg_total = np.mean(error_avgs)
    end_time = time.time()
    print("Execution time for trial: {:.2f} seconds. Hours tried: {}".format(end_time - start_time, hours))
    return error_avg_total


budget = 30
study = optuna.create_study(direction = "minimize")   # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=2)
study.optimize(objective, n_trials = budget)
print("The best set of parameters is: ", study.best_params)
print("The rmse score computed with inner-evaluation is:", study.best_value)
print(f"The optimal value is reached in {study.best_trial.number} iterations")



#%% PREDICTION ---------------------------------------

#Trial 12 finished with value: 186.4378712323999 and parameters: {'n_estimators': 806, 'max_depth': 38, 'min_samples_split': 5}
n_estimators = 806
max_depth = 38
min_samples_split = 5
n_cores = multiprocessing.cpu_count()

models = {}
for hour in range(24):
    
    X_train_hour = X_train.loc[X_train.index.hour == hour]
    Y_train_hour = Y_train.loc[Y_train.index.hour == hour]
    model = RandomForestRegressor(random_state = 100482712,
                                     n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split,
                                     n_jobs = n_cores)
    model.fit(X_train_hour,Y_train_hour)
    models[f"hour{hour}"] = model


file_name = "E:/Temporal/08_RF_24models.pickle"
with open(file_name, "wb") as f:  # Save model in HDD 
    #pickle.dump(models, f)
with open(file_name, "rb") as f:  # Read model from HDD
    models = pickle.load(f)

# Importance of features
n_show = 15
hours = ['hour{:01d}'.format(i) for i in range(24)]

fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(20, 24))
for i, ax_row in enumerate(axs):
    for j, ax_col in enumerate(ax_row):
        model = models[hours[i*4+j]]
        importances = model.feature_importances_
        feature_names = X_train.columns
        sorted_idx = importances.argsort()[::-1]
        sorted_importances = importances[sorted_idx][:n_show]
        sorted_feature_names = feature_names[sorted_idx][:n_show]
        ax_col.bar(range(n_show), sorted_importances)
        ax_col.set_xticks(range(n_show))
        ax_col.set_xticklabels(sorted_feature_names, rotation=90)
        ax_col.set_title(hours[i*4+j])      
plt.tight_layout()
plt.show()

# Predict
Y_pred = pd.DataFrame()
for hour in range(24):
    print(hour)
    X_test_hour = X_test.loc[X_test.index.hour == hour]
    model = models[f"hour{hour}"]
    Y_pred_hour = model.predict(X_test_hour)
    Y_pred_hour = pd.DataFrame(Y_pred_hour, index=X_test_hour.index, columns=Y_test.columns)
    Y_pred = pd.concat([Y_pred, Y_pred_hour])
Y_pred = Y_pred.sort_index()

#np.savetxt('Data/Preds/08_RF_24models_preds.csv', Y_pred, delimiter=',')
Y_pred = np.loadtxt("Data/Preds/08_RF_24models_preds.csv", delimiter=",")
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


plotPreds(Y_test, Y_pred, n_days=4, plotLag24=True)


import TFM_functions as tfm
stats, mae, mse = tfm.errors_analysis(Y_test, Y_pred, totals=True)

X_test=X_test.iloc[0:365*24]
Y_test=Y_test.iloc[0:365*24]
