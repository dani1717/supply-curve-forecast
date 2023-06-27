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
import seaborn as sns

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



#%% SERIES DISTRIBUTION

plt.figure(figsize=(12, 6))
df = data_train.iloc[:,0:50].melt(var_name='Q', value_name='MWh')
ax = sns.violinplot(x='Q', y='MWh', data=df, main='Series distribution')
_ = ax.set_xticklabels(data_train.iloc[:,0:50].keys(), rotation=90)

del df


#%% Seasonal decomposition, ACF and PACF

serie = 'Q27'
start = '2014-10-01'
end = '2014-10-29'

# Seasonal decomposition
dani.TS_seasonal_decompose(data_train[[serie]], from_date=start, to_date=end)

# ACF, PACF
dani.TS_plots(data_train[[serie]],lags_to_show=60, from_date=start, to_date=end)

# ACF, PACF of one hour of one series
hour = 11
data_train_hour = data_train.loc[data_train.index.hour == hour]
dani.TS_plots(data_train_hour[[serie]],lags_to_show=60, from_date=start, to_date=end)


#%% Study with diffs 1

TS =data_train[[serie]]
TS_diff1 = (TS - TS.shift(1)).dropna()


# Seasonal decomposition
dani.TS_seasonal_decompose(TS_diff1, from_date=start, to_date=end)

# ACF, PACF
dani.TS_plots(TS_diff1,60, from_date=start, to_date=end)

#%%

serie = 'Q30'
hour=11

data_short=data_train.iloc[0:365*24,:]
data_short_hour = data_short.loc[data_short.index.hour == hour]

fig, ax = plt.subplots(figsize=(15, 6))

palette = plt.get_cmap('hsv', 24)
# All hours
sns.lineplot(data_short.index.date, data_short[serie], hue=data_short.index.hour, palette=palette)
# One hour
sns.lineplot(data_short_hour.index.date, data_short_hour[serie], hue=data_short_hour.index.hour, palette=palette)

ax.set_title('Seasonal plot of Salinity', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Salinity Surface', fontsize = 16, fontdict=dict(weight='bold'))





















