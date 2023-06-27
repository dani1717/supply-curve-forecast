#%% IMPORTS --------------------------------------------------------------------
import pandas as pd
import numpy as np
import csv
import random
import datetime
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
import shap  # It's necessary to install numPy 1.21

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
    prefix_2weekEvol = 'sameHourEvol_pc'
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

period_in_months = None
train_years = None
split_date = '2020-12-31'
X_train, X_test, Y_train, Y_test = split_train_test(split_date, test_months=period_in_months, train_years=train_years)
    
# Scale
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

# Train model
print('    - Training model')
params = {'learning_rate': 0.2482427153031457, 'max_iter': 148, 'max_leaf_nodes': 80, 'max_depth': 2, 'min_samples_leaf': 60}
model_HGB = HistGradientBoostingRegressor(random_state=100482712, **params)
model = MultiOutputRegressor(model_HGB)
model.fit(X_train, Y_train)

# Predict
print('    - Predicting and measuring error')
Y_pred = model.predict(X_test)
Y_pred = pd.DataFrame(Y_pred)
Y_pred.index = Y_test.index
Y_pred.columns = Y_test.columns

explainer = shap.Explainer(model.estimators_[10])
shap_values = explainer(X_test)
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)

col = 'w1'
col_num = X_test.columns.get_loc(col)
shaps_vs = shap_values.values[:,col_num]
plt.scatter(X_test.loc[:, 'w1'], shaps_vs)
plt.xlabel(col)
plt.ylabel('shap value ('+col+')')
plt.title(col + " dependence plot")
plt.show()


shap.plots.scatter(shap_values[:,w1])




#%% Dataframe with global results

# Results for each series
shap_values_list = []
for estimator in model.estimators_:
    explainer = shap.Explainer(estimator)
    shap_values = explainer(X_test)
    shap_values_abs = np.abs(shap_values.values)  # Obtener los valores absolutos
    shap_values_mean = shap_values_abs.mean(0)  # Calcular la media de los valores absolutos
    shap_values_list.append(shap_values_mean)
df_shap_values = pd.DataFrame(shap_values_list, columns=X_test.columns)

#Reorganize dataframe
new_columns = ['lag24_pc1', 'mDiario_price', 'sameHourEvol_pc1', 'gas_price', 'w1']
df_selected = df_shap_values[new_columns]
df_other = df_shap_values.drop(columns=new_columns)
df_shap_values['other'] = df_other.sum(axis=1)
new_columns.append('other')
df_shap_values = df_shap_values[new_columns]
df_shap_values.index = df_shap_values.index +1 

# Plot it
import matplotlib.pyplot as plt
df_subset = df_shap_values.iloc[:, :5]
colors = ['blue', 'green', 'red', 'orange', 'purple']
labels = df_subset.columns
plt.figure(figsize=(10, 6))
for col, color in zip(df_subset.columns, colors):
    plt.plot(df_subset.index, df_subset[col], color=color, label=col)
plt.title("Shap values of main features for all Q's")
plt.xlabel('Q')
plt.ylabel('Shap value (mean)')
plt.legend()
plt.show()