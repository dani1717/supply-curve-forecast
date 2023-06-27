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

#%%

errors_lastYear = pd.read_csv('Data/Other/15_Errors_1M_weigthslastYear.csv', parse_dates=['start_date'])
errors_sameMonth = pd.read_csv('Data/Other/15_Errors_1M_weigthsSameMonth.csv', parse_dates=['start_date'])
errors_noWeights = pd.read_csv('Data/Other/15_Errors_1M.csv', parse_dates=['start_date'])

describe_lastYear = errors_lastYear['MAE'].describe().to_frame().transpose()
describe_sameMonth = errors_sameMonth['MAE'].describe().to_frame().transpose()
describe_noWeights = errors_noWeights['MAE'].describe().to_frame().transpose()
MAE_df = pd.concat([describe_lastYear, describe_sameMonth, describe_noWeights])
MAE_df.index = ['lastYear', 'sameMonth', 'noWeights']

describe_lastYear_rmse = errors_lastYear['RMSE'].describe().to_frame().transpose()
describe_sameMonth_rmse = errors_sameMonth['RMSE'].describe().to_frame().transpose()
describe_noWeights_rmse = errors_noWeights['RMSE'].describe().to_frame().transpose()
RMSE_df = pd.concat([describe_lastYear_rmse, describe_sameMonth_rmse, describe_noWeights_rmse])
RMSE_df.index = ['lastYear', 'sameMonth', 'noWeights']

# Crear una nueva figura y ejes
fig, ax1 = plt.subplots(dpi=300)

# Graficar las líneas para MAE de errors_lastYear
ax1.plot(errors_lastYear["start_date"], errors_lastYear["MAE"], color="blue", label="MAE - last year")

# Graficar las líneas para MAE de errors_sameMonth
ax1.plot(errors_sameMonth["start_date"], errors_sameMonth["MAE"], color="green", label="MAE - same month")

# Graficar las líneas para MAE de errors_noWeights
ax1.plot(errors_noWeights["start_date"], errors_noWeights["MAE"], color="purple", label="MAE - no weights")

# Graficar las líneas para MAE_24 de errors_lastYear
#ax1.plot(errors_lastYear["start_date"], errors_lastYear["MAE_24"], color="cyan", linestyle="--", label="MAE_24 - last year")

# Establecer etiquetas y colores para el primer eje y
ax1.set_xlabel("start_date")
ax1.set_ylabel("MAE")
ax1.tick_params(axis="y")

# Crear un segundo eje y para las columnas RMSE
ax2 = ax1.twinx()

# Graficar las líneas para RMSE de errors_lastYear
ax2.plot(errors_lastYear["start_date"], errors_lastYear["RMSE"], color="red", label="RMSE - last year")

# Graficar las líneas para RMSE de errors_sameMonth
ax2.plot(errors_sameMonth["start_date"], errors_sameMonth["RMSE"], color="orange", label="RMSE - same month")

# Graficar las líneas para RMSE de errors_noWeights
ax2.plot(errors_noWeights["start_date"], errors_noWeights["RMSE"], color="pink", label="RMSE - no weights")

# Graficar las líneas para RMSE_24 de errors_lastYear
#ax2.plot(errors_lastYear["start_date"], errors_lastYear["RMSE_24"], color="magenta", linestyle="--", label="RMSE_24 - last year")

# Establecer etiquetas y colores para el segundo eje y
ax2.set_ylabel("RMSE")
ax2.tick_params(axis="y")

# Combinar las líneas y etiquetas de ambos ejes
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]

x_ticks = errors_lastYear["start_date"][::len(errors_lastYear["start_date"]) // 12]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks.dt.date, rotation=90, ha="center")

# Establecer título y leyenda
ax1.set_title("Errors for monthly fit with different weights")
ax1.legend(lines, labels, loc="upper left", fontsize='small')
plt.show()


#%%

stats_MAE = pd.DataFrame({
    'lastYear_mean': [errors_lastYear['MAE'].mean()],
    'lastYear_std': [errors_lastYear['MAE'].std()],
    'lastYear_median': [errors_lastYear['MAE'].median()],
    'sameMonth_mean': [errors_sameMonth['MAE'].mean()],
    'sameMonth_std': [errors_sameMonth['MAE'].std()],
    'sameMonth_median': [errors_sameMonth['MAE'].median()],
    'noWeights_mean': [errors_noWeights['MAE'].mean()],
    'noWeights_std': [errors_noWeights['MAE'].std()],
    'noWeights_median': [errors_noWeights['MAE'].median()]
}, index=['MAE'])

stats_RMSE = pd.DataFrame({
    'lastYear_mean': [errors_lastYear['RMSE'].mean()],
    'lastYear_std': [errors_lastYear['RMSE'].std()],
    'lastYear_median': [errors_lastYear['RMSE'].median()],
    'sameMonth_mean': [errors_sameMonth['RMSE'].mean()],
    'sameMonth_std': [errors_sameMonth['RMSE'].std()],
    'sameMonth_median': [errors_sameMonth['RMSE'].median()],
    'noWeights_mean': [errors_noWeights['RMSE'].mean()],
    'noWeights_std': [errors_noWeights['RMSE'].std()],
    'noWeights_median': [errors_noWeights['RMSE'].median()]
}, index=['RMSE'])