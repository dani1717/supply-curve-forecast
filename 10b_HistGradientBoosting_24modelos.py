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

Y_test_unscaled = Y_test.copy()

#X_test = X_test.iloc[0:365*24,:]
#Y_test = Y_test.iloc[0:365*24,:]


#%% NORMALIZE DATA

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

# Scale X
scaler2 = StandardScaler()
cols_to_scale = ['mDiario_Q', 'mDiario_price', 'r1', 'r2', 'w1', 'w2', 'w3', 'w4', 'w5',
       'w6', 'w7', 'w8', 'w9', 'w10', 'gas_price', 'day_of_month',
       'lag24_pc1', 'lag24_pc2', 'lag24_pc3', 'lag24_pc4', 'lag24_pc5',
       '2weeksEvol_1', '2weeksEvol_2', '2weeksEvol_3', '2weeksEvol_4']
X_train[cols_to_scale] = scaler2.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler2.transform(X_test[cols_to_scale])

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
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
    max_iter = trial.suggest_int("max_iter", 50, 200)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 30)
    max_depth = trial.suggest_int("max_depth", 2, 60)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 10)

    start_time = time.time()
    error_avgs = []
    hours = [1,3,5,7,9,11,13,15,17,19,21,23]
    for hour in hours:
        # We calculate the error using the model for different hours and later return the average 
        X_train_hour = X_train.loc[X_train.index.hour == hour]
        Y_train_hour = Y_train.loc[Y_train.index.hour == hour]
        model = HistGradientBoostingRegressor(random_state=100482712,learning_rate=learning_rate,
                                              max_iter=max_iter,max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
        multioutput_model = MultiOutputRegressor(model)
        error_avg = np.sqrt(-np.mean(cross_val_score(multioutput_model,X_train_hour.values,Y_train_hour.values,
                    cv=tscv,scoring="neg_mean_squared_error",error_score="raise",n_jobs=n_cores)))
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

#Trial 1 finished with value: 0.6961756456231375 and parameters: {'learning_rate': 0.09037431981272596, 'max_iter': 103, 'max_leaf_nodes': 12, 'max_depth': 17, 'min_samples_leaf': 5}
learning_rate = 0.09037431981272596
max_iter = 103
max_leaf_nodes = 12
max_depth = 17
min_samples_leaf = 5

models = {}
for hour in range(24):
    
    X_train_hour = X_train.loc[X_train.index.hour == hour]
    Y_train_hour = Y_train.loc[Y_train.index.hour == hour]
    model_HGB = HistGradientBoostingRegressor(random_state=100482712,learning_rate=learning_rate,
                                          max_iter=max_iter,max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    model = MultiOutputRegressor(model_HGB)
    model.fit(X_train_hour,Y_train_hour)
    models[f"hour{hour}"] = model


# Predict
Y_pred = pd.DataFrame()
for hour in range(24):
    print('Predicting hour',hour)
    X_test_hour = X_test.loc[X_test.index.hour == hour]
    model = models[f"hour{hour}"]
    Y_pred_hour = model.predict(X_test_hour)
    Y_pred_hour = pd.DataFrame(Y_pred_hour, index=X_test_hour.index, columns=Y_test.columns)
    Y_pred = pd.concat([Y_pred, Y_pred_hour])
Y_pred = Y_pred.sort_index()

Y_pred = pd.DataFrame(Y_pred)
Y_pred.index = Y_test.index
Y_pred.columns = Y_test.columns

# Unscale
Y_pred_unscaled = pd.DataFrame(scaler.inverse_transform(Y_pred))
Y_pred_unscaled.columns = Y_test.columns
Y_pred_unscaled.index = Y_test.index

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

plotPreds(Y_test_unscaled, Y_pred_unscaled, n_days=2, plotLag24=True)

import TFM_functions as tfm
stats, mae, mse = tfm.errors_analysis(Y_test_unscaled, Y_pred_unscaled, totals=True)


Y_pred_unscaled2=Y_pred_unscaled.iloc[0:365*24]
Y_test_unscaled2=Y_test_unscaled.iloc[0:365*24]
stats, mae, mse = tfm.errors_analysis(Y_test_unscaled2, Y_pred_unscaled2, totals=True)


# Check errors by hour
for hour in range(24):
    Y_pred_hour = Y_pred_unscaled.loc[Y_pred_unscaled.index.hour == hour]
    Y_test_hour = Y_test_unscaled.loc[Y_test_unscaled.index.hour == hour]
    stats, e_mae, e_rmse = tfm.errors_analysis(Y_test_hour, Y_pred_hour, totals=True)
    print('Errors for hour:',hour,'\n   MAE:', e_mae,'   RMSE:',e_rmse)

'''

Errors for hour: 0 
   MAE: 157.17220787097273    RMSE: 201.70098516897787
Errors for hour: 1 
   MAE: 147.23280214977    RMSE: 187.80566393390544
Errors for hour: 2 
   MAE: 146.511180087201    RMSE: 186.97874464760645
Errors for hour: 3 
   MAE: 145.75294129855163    RMSE: 186.1824028793809
Errors for hour: 4 
   MAE: 147.12154184729798    RMSE: 186.94551151676825
Errors for hour: 5 
   MAE: 155.5331773122134    RMSE: 198.69072319487995
Errors for hour: 6 
   MAE: 178.41342463697592    RMSE: 227.8330062624908
Errors for hour: 7 
   MAE: 189.4192237617529    RMSE: 241.06561261664237
Errors for hour: 8 
   MAE: 197.75385302033206    RMSE: 249.67114033065312
Errors for hour: 9 
   MAE: 186.8555762910861    RMSE: 235.99295535402786
Errors for hour: 10 
   MAE: 164.22236703957878    RMSE: 211.5356740781305
Errors for hour: 11 
   MAE: 153.40048131715517    RMSE: 196.25818023665673
Errors for hour: 12 
   MAE: 153.1880067783396    RMSE: 197.44710020976657
Errors for hour: 13 
   MAE: 150.12238343958313    RMSE: 193.13946581418386
Errors for hour: 14 
   MAE: 155.3334564877527    RMSE: 199.75629797036612
Errors for hour: 15 
   MAE: 152.98704127164413    RMSE: 197.87319845299348
Errors for hour: 16 
   MAE: 145.27591672371142    RMSE: 186.22006931154718
Errors for hour: 17 
   MAE: 152.77798208119583    RMSE: 196.3053378715608
Errors for hour: 18 
   MAE: 163.6596728044788    RMSE: 211.62217609795414
Errors for hour: 19 
   MAE: 156.90279530983383    RMSE: 200.82467401731074
Errors for hour: 20 
   MAE: 158.89816819487535    RMSE: 204.88972212068558
Errors for hour: 21 
   MAE: 154.59527841861322    RMSE: 196.64008367258887
Errors for hour: 22 
   MAE: 159.66204663559532    RMSE: 201.213728994993
Errors for hour: 23 
   MAE: 150.95347840356635    RMSE: 191.93091753048205


'''







