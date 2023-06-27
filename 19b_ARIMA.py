import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
import random
from pmdarima import auto_arima

#%% LOAD DATA ------------------------------------------------------

data_original = pd.read_csv("Data/Other/15_fullData.csv")
data = data_original.iloc[:, 0:51]
data['datetime'] = pd.to_datetime(data['datetime'], format="%Y-%m-%d %H:%M:%S", utc=True)
data = pd.DataFrame(data)
data.set_index('datetime', inplace=True)

#%% PCA ------------------------------------------------------------

split_date = pd.to_datetime("2019-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S", utc=True)
data_train = data[data.index < split_date].dropna()
pca = PCA()
pca.fit(data_train)
n_comp = 2  # explained variance > 95%
Y = np.dot(data, pca.components_[:n_comp].T)
Y = pd.DataFrame(Y)
Y.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S", utc=True)

#%% ARIMA ------------------------------------------------

training_window_size = 30  # in days

p = 1
d = 0
q = 0
P = 2
D = 1
Q = 0

n_samples = 35
split_date = pd.to_datetime("2019-01-01 01:00:00", format="%Y-%m-%d %H:%M:%S", utc=True)
end_date = pd.to_datetime("2021-12-28", format="%Y-%m-%d", utc=True)
split_dates = []
for _ in range(n_samples):
    random_date = split_date + timedelta(days=random.randint(0, (end_date - split_date).days))
    split_dates.append(random_date)
split_dates.sort()

predictions1 = []
predictions2 = []
for split_date in split_dates:
    split_date_row = Y.index.get_loc(split_date)
    start_date = split_date - timedelta(days=training_window_size)
    start_date_row = Y.index.get_loc(start_date)
    serie1 = Y.iloc[start_date_row:(split_date_row-1), 0]
    serie2 = Y.iloc[start_date_row:(split_date_row-1), 1]
    serie1.index.freq = 'H'
    serie2.index.freq = 'H'
    print('Predicting for', split_date)
    model1 = auto_arima(serie1, start_q=0, max_q=0, d=0, start_Q=0, max_Q=0, D=1, seasonal=True, m=24, trace=False, njobs=4, maxiter=8)
    model2 = auto_arima(serie2, start_q=0, max_q=0, d=0, start_Q=0, max_Q=0, D=1, seasonal=True, m=24, trace=False, njobs=4, maxiter=8)
    prediction1 = model1.predict(n_periods=24)
    prediction2 = model2.predict(n_periods=24)
    predictions1.append(prediction1)
    predictions2.append(prediction2)
predictions1 = pd.concat(predictions1)
predictions2 = pd.concat(predictions2)
predictions = pd.concat([predictions1, predictions2], axis=1)
predictions.to_csv("Data/19b_predictions.csv", index=True, sep=';')

predictions = pd.read_csv("Data/19b_predictions.csv", sep=';')
predictions.set_index(predictions.columns[0], inplace=True)
predictions.index = pd.to_datetime(predictions.index)

preds_50 = np.dot(predictions, pca.components_[:n_comp])
preds_50 = pd.DataFrame(preds_50, index=predictions.index, columns=data.columns)
preds_50 = preds_50[~preds_50.index.duplicated(keep='first')]

real_values = data.loc[preds_50.index.intersection(data.index)]

import TFM_functions as tfm
stats, e_mae, e_rmse = tfm.errors_analysis(real_values, preds_50, totals=True)

real_values_2019 = real_values[real_values.index.year == 2019]
preds_50_2019 = preds_50[preds_50.index.year == 2019]
stats, e_mae19, e_rmse19 = tfm.errors_analysis(real_values_2019, preds_50_2019, totals=True)
