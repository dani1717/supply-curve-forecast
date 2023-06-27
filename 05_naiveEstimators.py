#%% IMPORTS --------------------------------------------------------------------
import pandas as pd
import TFM_functions as tfm

#%% LOAD DATA --------------------------------------------------------------------
folder = 'Data/'
files = {'approxs_subir': folder + "approxs_subir.csv", 
         'approxs_bajar': 'folder + "approxs_bajar.csv"', 
         'p': folder + "p.csv",
         'weather_data': folder + 'Weather data/weather_data_historical.csv',
         'gas_prices': folder + 'Gas prices/gas_prices.csv'}

data = pd.read_csv(files['approxs_subir'])

# Create some dummies like month, week, quarter
data.insert(0, 'datetime',  pd.to_datetime(data['date'], format='%Y%m%d'))
data['datetime'] = data['datetime'] + pd.to_timedelta(data['hour'] - 1, unit='h')
data.drop(['date', 'hour', 'weekday'], axis=1, inplace=True)
data.set_index('datetime', inplace=True)


#%% SPLIT TRAIN & TEST ------------------------------------------------------------

split_date = '2019-01-01'  # This day and further will be for testing

data_train = data.loc[data.index < split_date]
data_test = data.loc[data.index >= split_date]

del data, files, folder, split_date

#%% ESTIMATION: PREVIOUS DAY VALUE

# Train set
preds_lag24 = data_train.shift(24).dropna()
real_values = data_train.loc[preds_lag24.index]
stats,mae_prevDay,mse_prevDay = tfm.errors_analysis(real_values, preds_lag24,totals=True)

# Test set
preds_lag24 = data_test.shift(24).dropna()
real_values = data_test.loc[preds_lag24.index]
stats,mae_prevDay,mse_prevDay = tfm.errors_analysis(real_values, preds_lag24,totals=True)

# Only 2019
data_2019 = data_test.iloc[0:365*24,:]
preds_lag24 = data_2019.shift(24).dropna()
real_values = data_2019.loc[preds_lag24.index]
stats,mae_prevDay,mse_prevDay = tfm.errors_analysis(real_values, preds_lag24,totals=True)


#%% #%% ESTIMATION: PREVIOUS WEEK VALUE

preds_prevWeek = data_train.shift(24*7).dropna()
real_values = data_train.loc[preds_prevWeek.index]

stats,mae_prevWeek,mse_prevWeek = tfm.errors_analysis(real_values, preds_prevWeek,totals=True)
