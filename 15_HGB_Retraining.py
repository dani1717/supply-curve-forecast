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

#%% LOAD ORIGINAL VALUES
# For Lag24 estimator comparison

# Load all data
data = pd.read_csv('Data/Other/15_fullData.csv', index_col='datetime')
data.index = pd.to_datetime(data.index)

# Extract Y
Y_all = data.iloc[:,:50].copy()

del data

#%% SPLIT TRAIN & TEST FUNCTION ------------------------------------------------------------

def split_train_test (split_date, test_months=None, train_years=None, n_days_back=15*7, n_allData_back=3):
    # Returns from data train and test dataframes. split_date is the first date for the test dataset
    # test set is test_months long. If it is none, it takes all the dataset except training part
    # If train_years=None the train set starts from the beginning. Otherwise it is train_years long finishing in split_date
    
    import warnings
    warnings.filterwarnings("ignore")
    
    data = pd.read_csv('Data/Other/15_fullData.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    
    if isinstance(split_date, str):
        split_date = pd.to_datetime(split_date)
        
        
    if train_years is None:
         data_train = data.loc[data.index < split_date]
    else:
         start_date = split_date - pd.DateOffset(years=train_years)
         data_train = data.loc[(data.index >= start_date) & (data.index < split_date)]
    
    if test_months==None:
        test_months = 999
    
    data_test = data.loc[(data.index >= split_date) & (data.index < split_date + pd.DateOffset(months=test_months))]
    
    #WEATHER DATA -----------------------------------------------------------------
    
    # Scale X
    scaler2 = StandardScaler()
    cols_to_scale = data_train.columns[data_train.columns.get_loc('Q50')+1:data_train.columns.get_loc('day_of_month')+1]
    data_train.loc[:,cols_to_scale] = scaler2.fit_transform(data_train.loc[:,cols_to_scale])
    data_test.loc[:,cols_to_scale] = scaler2.transform(data_test.loc[:,cols_to_scale])
    
    # DIMENSION REDUCTION OF RADIATION AND WIND
    
    # PCA radiation
    cols = data_train.columns[data_train.columns.get_loc('r_ACor'):data_train.columns.get_loc('r_Zara')+1]
    #dani.PCA_analysis(data_train[cols])
    n_comps = 2
    pca = PCA().fit(data_train[cols])
    data_train = dani.substitute_with_PCA(data_train,cols,n_comps,pca,prefix='r')
    data_test = dani.substitute_with_PCA(data_test, cols, n_comps, pca, prefix='r')
    
    # PCA wind
    cols = data_train.columns[data_train.columns.get_loc('w_ACor'):data_train.columns.get_loc('w_Zara')+1]
    #dani.PCA_analysis(data_train[cols])
    n_comps = 10
    pca = PCA().fit(data_train[cols])
    data_train = dani.substitute_with_PCA(data_train,cols,n_comps,pca,prefix='w')
    data_test = dani.substitute_with_PCA(data_test,cols,n_comps,pca,prefix='w')
    
    # Scale again (PCs are not scaled)
    scaler2 = StandardScaler()
    cols_to_scale = data_train.columns[data_train.columns.get_loc('r1'):data_train.columns.get_loc('w'+ str(n_comps))+1]
    data_train[cols_to_scale] = scaler2.fit_transform(data_train[cols_to_scale])
    data_test[cols_to_scale] = scaler2.transform(data_test[cols_to_scale])
    
    # ADDING LAGS -----------------------------------------------------------------
    
    # lag24 for Q_i
    data_all = pd.concat([data_train, data_test])
    cols = data_all.columns[data_all.columns.get_loc('Q1'):data_all.columns.get_loc('Q50')+1]
    lag_cols = [col+'_lag24' for col in cols]
    lags = data_all[cols].shift(24).rename(columns=dict(zip(cols, lag_cols)))
    data_all = pd.concat([data_all, lags], axis=1).dropna()
    data_train = data_all.loc[:data_train.index.max()].dropna()
    data_test = data_all.loc[data_test.index.min():].dropna()
    
    # Reduce dimension of lags_24
    cols = data_train.columns[data_train.columns.get_loc('Q1_lag24'):data_train.columns.get_loc('Q50_lag24')+1]
    #dani.PCA_analysis(data_train[cols])
    n_comps = 5
    prefix_lags24pcs = 'lag24_pc'
    pca = PCA().fit(data_train[cols])
    data_train = dani.substitute_with_PCA(data_train,cols,n_comps,pca,prefix=prefix_lags24pcs)
    data_test = dani.substitute_with_PCA(data_test,cols,n_comps,pca,prefix=prefix_lags24pcs)
    
    #---------------------------------------

    # Add lags for the same hour last months
    n_days = n_days_back
    data_all = pd.concat([data_train, data_test])
    for day in range(1, n_days+1):
        col_name = 'lag24_pc1_lagDays' + str(day)
        data_all[col_name] = data_all[prefix_lags24pcs + '1'].shift(day*24)
    data_train = data_all.loc[:data_train.index.max()].dropna()
    data_test = data_all.loc[data_test.index.min():].dropna()

    # Reduce dimensionality
    cols = data_train.columns[data_train.columns.get_loc('lag24_pc1_lagDays1'):data_train.columns.get_loc('lag24_pc1_lagDays'+str(n_days))+1]
    #dani.PCA_analysis(data_train[cols])
    n_comps = 15
    prefix_2weekEvol = '2weeksEvol_'
    pca = PCA().fit(data_train[cols])
    data_train = dani.substitute_with_PCA(data_train,cols,n_comps,pca,prefix=prefix_2weekEvol)
    data_test = dani.substitute_with_PCA(data_test,cols,n_comps,pca,prefix=prefix_2weekEvol)

    #---------------------------------------

    # Add lags for all hours for some previous days
    n_days = n_allData_back
    data_all = pd.concat([data_train, data_test])
    prefix_allHours = 'hours_lag'
    for hour in range(25, n_days*24+1):
        col_name = prefix_allHours + str(hour)
        data_all[col_name] = data_all[prefix_lags24pcs + '1'].shift(hour)
    data_train = data_all.loc[:data_train.index.max()].dropna()
    data_test = data_all.loc[data_test.index.min():].dropna()

    # Reduce dimensionality
    cols = data_train.columns[data_train.columns.get_loc(prefix_allHours + str(25)):data_train.columns.get_loc(prefix_allHours + str(n_days*24))+1]
    #dani.PCA_analysis(data_train[cols])
    n_comps = 8
    pca = PCA().fit(data_train[cols])
    data_train = dani.substitute_with_PCA(data_train,cols,n_comps,pca,prefix=prefix_allHours)
    data_test = dani.substitute_with_PCA(data_test,cols,n_comps,pca,prefix=prefix_allHours)

    #---------------------------------------

    # Scale again (last PCs are not scaled)
    scaler2 = StandardScaler()
    cols_to_scale = data_train.columns[data_train.columns.get_loc(prefix_lags24pcs+'1'):data_train.columns.get_loc(prefix_allHours+str(n_comps))+1]
    data_train[cols_to_scale] = scaler2.fit_transform(data_train[cols_to_scale])
    data_test[cols_to_scale] = scaler2.transform(data_test[cols_to_scale])

    stats = data_train.describe().T[['mean', 'std']]

    del col_name, cols, cols_to_scale, data_all, day, lag_cols, lags, n_comps, scaler2, stats
    del prefix_2weekEvol, prefix_lags24pcs
    
    # EXPORT
    
    cols_Q = data_train.columns[data_train.columns.get_loc('Q1'):data_train.columns.get_loc('Q50')+1]
    X_train = data_train.drop(cols_Q, axis=1)
    Y_train = data_train[cols_Q]
    X_test = data_test.drop(cols_Q, axis=1)
    Y_test = data_test[cols_Q]

    return X_train, X_test, Y_train, Y_test


#%% LOOP

period_in_months = 1
train_years = None

start_date = '2019-01-01'
end_date = '2021-12-31'
split_dates = pd.date_range(start=start_date, end=end_date, freq=f'{period_in_months}MS')

Y_pred_unscaled_all = pd.DataFrame()
errors = pd.DataFrame(columns=["start_date", "MAE", "RMSE"])
for split_date in split_dates:
    
    print('Predicting for the next',period_in_months,'months starting from',split_date)
    
    # Generate train and test sets
    print('    - Spliting dataset')
    X_train, X_test, Y_train, Y_test = split_train_test(split_date, test_months=period_in_months, train_years=train_years, 
                                                        n_days_back=12*7, n_allData_back=4)
    
    # Save
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
    
    # Weights
    weights_lastYear = np.ones(len(X_train))
    weights_lastYear[-365*24:] = 2

    month = X_test.iloc[0].name.month
    weights_sameMonth = np.ones(len(X_train))
    weights_sameMonth[X_train.index.month == month] = 2
    
    
    # Train model
    print('    - Training model')
    params = {'learning_rate': 0.2482427153031457, 'max_iter': 148, 'max_leaf_nodes': 80, 'max_depth': 2, 'min_samples_leaf': 60}
    model_HGB = HistGradientBoostingRegressor(random_state=100482712, **params)
    model = MultiOutputRegressor(model_HGB)
    model.fit(X_train, Y_train, sample_weight=weights_lastYear)
    
    # Predict
    print('    - Predicting and measuring error')
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.index = Y_test.index
    Y_pred.columns = Y_test.columns

    # Unscale, force monotonicity and measure error
    Y_pred_unscaled = pd.DataFrame(scaler.inverse_transform(Y_pred))
    Y_pred_unscaled.columns = Y_test.columns
    Y_pred_unscaled.index = Y_test.index
    Y_pred_unscaled = tfm.restore_monotonicity(Y_pred_unscaled, max_iter=50)
    _, e_mae, e_rmse = tfm.errors_analysis(Y_test_unscaled, Y_pred_unscaled, totals=True)
    
    # Calculate lag24 estimator and measure error
    Y_pred_lag24 = Y_all.loc[Y_test.index - pd.DateOffset(hours=24),:]
    Y_pred_lag24.index = Y_test.index
    _, e_mae24, e_rmse24 = tfm.errors_analysis(Y_test_unscaled, Y_pred_lag24, totals=True)
    
    # Add preds and errors to dataframe
    if Y_pred_unscaled_all.empty:
        Y_pred_unscaled_all = Y_pred_unscaled
    else:
        Y_pred_unscaled_all = pd.concat([Y_pred_unscaled_all, Y_pred_unscaled], axis=0)
    new_row = {"start_date": split_date, "MAE": e_mae, "RMSE": e_rmse, 'MAE_24': e_mae24, 'RMSE_24': e_rmse24}
    errors = errors.append(new_row, ignore_index=True)

#%% PLOT RESULTS

Y_pred_unscaled_all.to_csv('Data/Other/15_Y_preds_all.csv')
errors.to_csv('Data/Other/15_Errors_1M_weigthslastYear.csv')

fig, ax1 = plt.subplots(dpi=300)
ax1.plot(errors["start_date"], errors["MAE"], color="blue")
ax1.plot(errors["start_date"], errors["MAE_24"], color="cyan", linestyle="--", label="lag24 estimator - MAE")  
ax1.set_xlabel("start_date")
ax1.set_ylabel("MAE", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(errors["start_date"], errors["RMSE"], color="red")
ax2.plot(errors["start_date"], errors["RMSE_24"], color="magenta", linestyle="--", label="lag24 estimator - RMSE")
ax2.set_ylabel("RMSE", color="red")
ax2.tick_params(axis="y", labelcolor="red")

lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
x_ticks = errors["start_date"][::len(errors["start_date"]) // 12]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks.dt.date, rotation=90, ha="center")

ax1.set_title("Errors for monthly fit - HGB")
ax1.legend(lines, labels, loc="upper right")
plt.show()

import winsound
winsound.Beep(1000, 1000)