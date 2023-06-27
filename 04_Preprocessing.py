#%% IMPORTS --------------------------------------------------------------------
import pandas as pd
import numpy as np
import csv
import random
import datetime
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pickle
import myFunctions_Dani as dani
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import time
import optuna
import multiprocessing

#%% LOAD DATA --------------------------------------------------------------------

folder = 'Data/'
files = {'approxs_subir': folder + "approxs_subir.csv", 
         'approxs_bajar': 'folder + "approxs_bajar.csv"', 
         'p': folder + "p.csv",
         'weather_data': folder + 'Weather data/weather_data_historical.csv',
         'gas_prices': folder + 'Gas prices/gas_prices.csv',
         'mDiario_Q': [folder + 'Mercado_diario/' + 'export_EnergiaAsignadaEnMercadoSPOTDiarioEspana_1.csv' , folder + 'Mercado_diario/' + 'export_EnergiaAsignadaEnMercadoSPOTDiarioEspana_2.csv',
                        folder + 'Mercado_diario/' + 'export_EnergiaAsignadaEnMercadoSPOTDiarioEspana_3.csv',folder + 'Mercado_diario/' + 'export_EnergiaAsignadaEnMercadoSPOTDiarioEspana_4.csv'],
         'mDiario_price': [folder + 'Mercado_diario/' + 'export_PrecioMercadoSPOTDiario_1.csv',folder + 'Mercado_diario/' + 'export_PrecioMercadoSPOTDiario_2.csv',
                           folder + 'Mercado_diario/' + 'export_PrecioMercadoSPOTDiario_3.csv',folder + 'Mercado_diario/' + 'export_PrecioMercadoSPOTDiario_4.csv']}

# Load files
curves = pd.read_csv(files['approxs_subir'])
weekdays = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
curves['weekday'] = curves['weekday'].map(weekdays)
weather_data = pd.read_csv(files['weather_data'])
gas_prices = pd.read_csv(files['gas_prices'])
mDiario_Q = pd.concat([pd.read_csv(file, sep=';') for file in files['mDiario_Q']], axis=0)
mDiario_Q['datetime'] = pd.to_datetime(mDiario_Q['datetime'])
mDiario_Q['datetime'] = mDiario_Q['datetime'].apply(datetime.datetime.replace, tzinfo=None)
mDiario_Q = mDiario_Q.rename(columns={"value": "mDiario_Q"})
mDiario_price = pd.concat([pd.read_csv(file, sep=';') for file in files['mDiario_price']], axis=0)
mDiario_price['datetime'] = pd.to_datetime(mDiario_price['datetime'])
mDiario_price['datetime'] = mDiario_price['datetime'].apply(datetime.datetime.replace, tzinfo=None)
mDiario_price = mDiario_price.rename(columns={"value": "mDiario_price"})


# Create datetime column
data = curves
data.insert(0, 'datetime',  pd.to_datetime(data['date'], format='%Y%m%d'))
data['datetime'] = data['datetime'] + pd.to_timedelta(data['hour'] - 1, unit='h')

# Merge
data = data.merge(mDiario_Q.loc[:,['datetime','mDiario_Q']], on=['datetime'], how='left')
data = data.merge(mDiario_price.loc[:,['datetime','mDiario_price']], on=['datetime'], how='left')
data['mDiario_price'].fillna(method='ffill', inplace=True)  # There is no price and Q for winter and summer hour change
data['mDiario_Q'].fillna(method='ffill', inplace=True)
data = data.merge(weather_data, on=['date', 'hour'], how='left')
data = data.merge(gas_prices, on=['date'], how='left')


# Create some dummies like month, week, quarter
data['day_of_month'] = data['datetime'].dt.day
dummies_hour = pd.get_dummies(data['datetime'].dt.hour, prefix='hour')
dummies_weekday = pd.get_dummies(data['weekday'], prefix='weekday')
dummies_month = pd.get_dummies(data['datetime'].dt.month, prefix='month')
dummies_quarter = pd.get_dummies(data['datetime'].dt.quarter, prefix='quarter')
data = pd.concat([data, dummies_hour, dummies_weekday, dummies_month, dummies_quarter], axis=1)
data.drop('weekday', axis=1, inplace=True)
data.set_index('datetime', inplace=True)

# Include holidays col
with open('Data/holidays.csv', newline='') as archivo_csv:
    reader = csv.reader(archivo_csv)
    holidays = [int(entero) for entero in next(reader)]
data['holiday'] = data['date'].apply(lambda x: 1 if x in holidays else 0)

data.drop('date', axis=1, inplace=True)
data.drop('hour', axis=1, inplace=True)
data = data.resample('H').mean()  # Some days have different gas prices, with this we take the average

data_original = data.copy()



#%% SPLIT TRAIN & TEST ------------------------------------------------------------

split_date = '2019-01-01'  # This day and further will be for testing

data_train = data.loc[data.index < split_date]
data_test = data.loc[data.index >= split_date]

del archivo_csv, curves, data, dummies_hour, dummies_month, dummies_quarter, dummies_weekday, mDiario_price, mDiario_Q
del files, folder, gas_prices, holidays, reader, split_date, weather_data, weekdays


#%% WEATHER DATA -----------------------------------------------------------------

data_train.columns.tolist()

'''
# Scale Q
scaler = StandardScaler()
cols_to_scale = data_train.columns[data_train.columns.get_loc('Q1'):data_train.columns.get_loc('Q50')+1]
data_train[cols_to_scale] = scaler.fit_transform(data_train[cols_to_scale])
data_test[cols_to_scale] = scaler.transform(data_test[cols_to_scale])
'''

# Scale others
scaler2 = StandardScaler()
cols_to_scale = data_train.columns[data_train.columns.get_loc('Q50')+1:data_train.columns.get_loc('day_of_month')+1]
data_train[cols_to_scale] = scaler2.fit_transform(data_train[cols_to_scale])
data_test[cols_to_scale] = scaler2.transform(data_test[cols_to_scale])

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

del cols, cols_to_scale, n_comps, scaler2


#%% ADDING LAGS -----------------------------------------------------------------

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


# Add lags for the same hour last two weeks
data_all = pd.concat([data_train, data_test])
for day in range(1, 15):
    col_name = 'lag24_pc1_lagDays' + str(day)
    data_all[col_name] = data_all[prefix_lags24pcs + '1'].shift(day*24)
data_train = data_all.loc[:data_train.index.max()].dropna()
data_test = data_all.loc[data_test.index.min():].dropna()

# Reduce dimensionality
cols = data_train.columns[data_train.columns.get_loc('lag24_pc1_lagDays1'):data_train.columns.get_loc('lag24_pc1_lagDays14')+1]
#dani.PCA_analysis(data_train[cols])
n_comps = 4
prefix_2weekEvol = '2weeksEvol_'
pca = PCA().fit(data_train[cols])
data_train = dani.substitute_with_PCA(data_train,cols,n_comps,pca,prefix=prefix_2weekEvol)
data_test = dani.substitute_with_PCA(data_test,cols,n_comps,pca,prefix=prefix_2weekEvol)

# Scale again (last PCs are not scaled)
scaler2 = StandardScaler()
cols_to_scale = data_train.columns[data_train.columns.get_loc(prefix_lags24pcs+'1'):data_train.columns.get_loc(prefix_2weekEvol+str(n_comps))+1]
data_train[cols_to_scale] = scaler2.fit_transform(data_train[cols_to_scale])
data_test[cols_to_scale] = scaler2.transform(data_test[cols_to_scale])

stats = data_train.describe().T[['mean', 'std']]

del col_name, cols, cols_to_scale, data_all, day, lag_cols, lags, n_comps, scaler2, stats
del prefix_2weekEvol, prefix_lags24pcs


#%% EXPORT

cols_Q = data_train.columns[data_train.columns.get_loc('Q1'):data_train.columns.get_loc('Q50')+1]
X_train = data_train.drop(cols_Q, axis=1)
Y_train = data_train[cols_Q]
X_test = data_test.drop(cols_Q, axis=1)
Y_test = data_test[cols_Q]

folder = 'Data/Preprocessed_inputs/'
sufix = ''
X_train.to_csv(folder + 'X_train' + sufix + '.csv')
X_test.to_csv(folder + 'X_test' + sufix + '.csv')
Y_train.to_csv(folder + 'Y_train' + sufix + '.csv')
Y_test.to_csv(folder + 'Y_test' + sufix + '.csv')

file_name = "Data/Other/04_Preprocessing_scaler.pickle"
with open(file_name, "wb") as f:
    pickle.dump(scaler, f)
file_name = "Data/Other/04_Preprocessing_data_original.pickle"
with open(file_name, "wb") as f:
    pickle.dump(data_original, f)
