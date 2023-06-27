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
import seaborn as sns
from matplotlib import ticker
from sklearn.metrics import mean_absolute_error

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

# Predictions
Y_test_preds = pd.read_csv('Data/Other/15_Y_preds_all.csv')
Y_test_preds['datetime'] = pd.to_datetime(Y_test_preds['datetime'])
Y_test_preds.set_index('datetime', inplace=True)

# Dates when requirement and offer don't match
noMatchDates = pd.read_csv('Data/noMatchDates_subir.csv')
noMatchDates.insert(0, 'datetime',  pd.to_datetime(noMatchDates['date'], format='%Y%m%d'))
noMatchDates['datetime'] = noMatchDates['datetime'] + pd.to_timedelta(noMatchDates['hour'] - 1, unit='h')

# Errors
errors = pd.read_csv('Data/Other/15_Errors_1M_weigthslastYear.csv', parse_dates=['start_date'])

# Final prices
finalPrices = pd.read_csv('Data/finalPrices_subir.csv', parse_dates=['date'])
finalPrices['date'] = pd.to_datetime(finalPrices['date']) + pd.to_timedelta(finalPrices['hour'] - 1, unit='h')
finalPrices = finalPrices.drop('hour', axis=1)
finalPrices.set_index('date', inplace=True)

# Requirements
requirements = pd.read_csv('Data/requirements_subir.csv', parse_dates=['date'])
requirements['date'] = pd.to_datetime(requirements['date']) + pd.to_timedelta(requirements['hour'] - 1, unit='h')
requirements = requirements.drop('hour', axis=1)
requirements.set_index('date', inplace=True)

# Grid of prices
p = pd.read_csv('Data/p.csv')['x'].tolist()

# Lag24 preds
file_name = "Data/Other/04_Preprocessing_data_original.pickle"
with open(file_name, "rb") as f:
    data_original = pickle.load(f)
del folder, file_name, 
Y_lag24 = data_original.iloc[:,:50].shift(24)
Y_lag24 = Y_lag24.loc[Y_test_preds.index]

# Covid data  https://cnecovid.isciii.es/covid19/#documentaci%C3%B3n-y-datos
covid = pd.read_csv('Data/casos_hosp_uci_def_sexo_edad_provres.csv', parse_dates=['fecha'])
covid = covid.iloc[:,3:]
covid = covid.groupby(covid['fecha'].dt.to_period('M')).sum()
covid = covid.loc[covid.index.year <= 2021]


#%% CHECK MONOTONICITY

wrong_rows = ~Y_test_preds.apply(lambda row: row.is_monotonic_increasing, axis=1)
wrong_rows = wrong_rows[wrong_rows].index
print('Non monotonic rows:',wrong_rows.shape[0])

#%% FILTER DATES WHEN REQ AND OFFER DON'T MATCH

mask = Y_test.index.isin(noMatchDates['datetime'])
Y_test = Y_test[~mask]

mask = Y_test_preds.index.isin(noMatchDates['datetime'])
Y_test_preds = Y_test_preds[~mask]

#%% MAE, RMSE

# Total MAE and RMSE
stats, e_mae, e_rmse = tfm.errors_analysis(Y_test, Y_test_preds, totals=True)

# Errors per month
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

# Improvement plot
mae_diff = -(errors["MAE"] - errors["MAE_24"])*100/errors["MAE"]
rmse_diff = -(errors["RMSE"] - errors["RMSE_24"])*100/errors["RMSE"]


fig, ax = plt.subplots(dpi=300)
ax.plot(errors["start_date"], mae_diff, color="blue", label="MAE difference")
ax.plot(errors["start_date"], rmse_diff, color="red", label="RMSE difference")
ax.set_xlabel("start_date")
ax.set_ylabel("Difference")
ax.set_title("Difference between MAEs and RMSEs per month - HGB")
ax.legend()
ax.grid(True)

x_ticks = errors["start_date"][::len(errors["start_date"]) // 12]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks.dt.date, rotation=90, ha="center")

plt.show()



#%% COMPUTE APPROXIMATION OF FINAL PRICES AND NAÏF ESTIMATORS

finalPrices_preds = []
finalPrices_preds_lag24 = []
for idx, row in Y_test_preds.iterrows():
    req = requirements.loc[idx]
    p_idx = Y_test_preds.loc[idx,].searchsorted(req)[0]
    finalPrices_preds.append(p[p_idx])
    p_idx = Y_lag24.loc[idx,].searchsorted(req)[0]
    if p_idx==50:
        finalPrices_preds_lag24.append(600)
    else:
        finalPrices_preds_lag24.append(p[p_idx])

finalPrices_lag24 = finalPrices['finalPrice'].shift(24)
finalPrices_lag24 = finalPrices_lag24.loc[Y_test_preds.index]

finalPrices_All = pd.DataFrame(index=Y_test_preds.index)
finalPrices_All['finalPrice'] = finalPrices['finalPrice']
finalPrices_All['pred'] = finalPrices_preds
finalPrices_All['finalPrice_lag24'] = finalPrices_lag24   # lag 24 of finalPrice
finalPrices_All['pred_lag24'] = finalPrices_preds_lag24   # intersection between requirement and lag24 supply curve

finalPrices_errors = finalPrices_All.iloc[:, 1:].sub(finalPrices_All.iloc[:, 0], axis=0).abs()

#%% RESULTS

# Table
finalPrices_errors.describe(percentiles=[.25, .5, .75, .9, .95, .99])
# 2021
finalPrices_errors.loc[finalPrices_errors.index.year == 2021].describe(percentiles=[.25, .5, .75, .9, .95, .99])

# Histograms
plt.hist(finalPrices_errors.loc[:,'pred'], bins=80)
plt.xlabel('€')
plt.ylabel('freq')
plt.title('finalPrice errors')
plt.show()

plt.hist(finalPrices_errors.loc[:,'finalPrice_lag24'], bins=80)
plt.xlabel('€')
plt.ylabel('freq')
plt.title('Naïf estimator errors')
plt.show()

# Violin plot
plt.figure(figsize=(8, 6), dpi=300)
sns.violinplot(data=finalPrices_errors.iloc[:,:2], inner="quartile")
plt.xlabel('Columnas')
plt.ylabel('Errores')
plt.title('Diagrama de Violín de los Errores')
plt.show()

# Worst prediction. (In the graph Real values is the approximation of the curve, not the original curve)
dateBad = finalPrices_errors['pred'].idxmax()
tfm.compareCurve(Y_test, Y_test_preds,plotLag24=True, reqs=requirements, date=dateBad)

date0 = datetime.datetime.strptime('2021-12-01 05:00:00', '%Y-%m-%d %H:%M:%S')
date0 = None
tfm.compareCurve(Y_test, Y_test_preds,plotLag24=True, reqs=requirements, date=date0)

start_date = datetime.datetime(2021, 1, 1)
end_date = datetime.datetime(2021, 12, 31, 23, 0, 0)
random_date = start_date + datetime.timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
random_date = random_date.replace(minute=0, second=0)
tfm.compareCurve(Y_test, Y_test_preds,plotLag24=True, reqs=requirements, date=random_date)

#%% PLOT BY MONTH

monthly_avg = finalPrices_errors.resample('M')['pred'].mean()
monthly_percentiles = finalPrices_errors.groupby(finalPrices_errors.index.to_period('M'))['pred'].quantile([0.05, 0.95])
monthly_lower = monthly_percentiles.xs(0.05, level=1)
monthly_upper = monthly_percentiles.xs(0.95, level=1)

plt.fill_between(monthly_avg.index.strftime('%b-%y'), monthly_lower.values, monthly_upper.values, alpha=0.3)
plt.plot(covid.index.strftime('%b-%y'), covid['num_casos'] / covid['num_casos'].max() * monthly_upper.max(), color='blue', linestyle='--', label='Number of COVID-19 cases (scaled)')
plt.plot(monthly_avg.index.strftime('%b-%y'), monthly_avg.values)

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.IndexLocator(base=3, offset=0))
plt.ylabel('Error (€/MWh)')
plt.title('Error in clearing price prediction per month')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('results_priceErrorByMonth_covid.png', dpi=300)
plt.show()


#%% TABLE BY YEAR

errors_byYear = finalPrices_errors.groupby(finalPrices_errors.index.year)['pred'].agg([
    ('mean', 'mean'),
    ('min', 'min'),
    ('q1', lambda x: np.percentile(x, 25)),
    ('q2', 'median'),
    ('q3', lambda x: np.percentile(x, 75)),
    ('p90', lambda x: np.percentile(x, 90)),
    ('p95', lambda x: np.percentile(x, 95)),
    ('p99', lambda x: np.percentile(x, 99)),
    ('max', 'max')
])

print(errors_byYear)

#%% SCATTERPLOT MAE-MAE WITH LAG24

# MAE betwwen each curve and lag24
mae_list_toLag24 = []
mae_list_toRealValue = []
for index in Y_test_preds.index:
    y_test_pred_row = Y_test_preds.loc[index]
    y_lag24_row = Y_lag24.loc[index]
    y_realValue_row = Y_test.loc[index]
    mae_lag24 = mean_absolute_error(y_test_pred_row, y_lag24_row)
    mae_realValue = mean_absolute_error(y_test_pred_row, y_realValue_row)
    mae_list_toLag24.append(mae_lag24)
    mae_list_toRealValue.append(mae_realValue)
mae_df = pd.DataFrame({'MAE_toLag24': mae_list_toLag24, 'MAE_toRealValue': mae_list_toRealValue}, index=Y_test_preds.index)
mae_median = np.median(mae_list_toRealValue)
#mae_df = mae_df[mae_df['MAE_toRealValue'] >= mae_median]


plt.scatter(mae_df['MAE_toRealValue'], mae_df['MAE_toLag24'], alpha=0.16, s=2)
plt.axvline(mae_median, color='blue', linestyle='--', label='Median')
plt.xlabel('MAE to real curve')
plt.ylabel('MAE to lag 24')
plt.title('MAE from predicted curve to real curve and to lag 24')
plt.legend()
plt.savefig('conclusions_scatterplot.png', dpi=300)
plt.show()

# Correlation
np.corrcoef(mae_list_toRealValue, mae_list_toLag24)[0, 1]

# Boxplot of MAE to lag24 by quartiles

mae_df_copy = mae_df.copy()
mae_df_copy['Group'] = pd.qcut(mae_df_copy['MAE_toRealValue'], q=4)
groups = mae_df_copy['Group'].unique()

data = []

for group in groups:
    data.append(mae_df_copy.loc[mae_df_copy['Group'] == group, 'MAE_toLag24'])

plt.boxplot(data, labels=groups, showfliers=False)
plt.xticks([])
plt.xlabel('MAE to real curve (Quartile groups)')
plt.ylabel('MAE to Lag24')
plt.title('Boxplot of MAE to Lag24 by groups')
plt.show()

#%% SCATTERPLOT ERROR-MAE

finalPrices_errors['Year'] = finalPrices_errors.index.year
mae_df['Year'] = mae_df.index.year


color_dict = {2019: 'red', 2020: 'yellow', 2021: 'green'}
scatter = plt.scatter(finalPrices_errors['pred'], mae_df['MAE_toRealValue'], alpha=0.5, s=1, c=finalPrices_errors.index.year.map(color_dict))
plt.xlabel('Error in clearing price prediction (€/MWh)')
plt.ylabel('MAE to real curve')
plt.title('Curve prediction and price error')
handles, labels = scatter.legend_elements()
legend_elements = []
for year in finalPrices_errors.index.year.unique():
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=str(year), markerfacecolor=color_dict[year], markersize=5))
plt.legend(handles=legend_elements, title='Year')
plt.savefig('conclusions_scatterplot.png', dpi=300)
plt.show()

#%% 

requirements_filtered = requirements.copy()
requirements_filtered = requirements_filtered.loc['2019-01-01':'2021-12-31']
requirements_filtered.index = pd.to_datetime(requirements_filtered.index)

data_2019 = requirements_filtered.loc[requirements_filtered.index.year == 2019][requirements_filtered.columns[0]]
data_2020 = requirements_filtered.loc[requirements_filtered.index.year == 2020][requirements_filtered.columns[0]]
data_2021 = requirements_filtered.loc[requirements_filtered.index.year == 2021][requirements_filtered.columns[0]]

values = [data_2019.tolist(), data_2020.tolist(), data_2021.tolist()]
box_labels = ['2019', '2020' ,'2021']

plt.boxplot(values)
plt.xticks(range(1, 4), box_labels)
plt.title('Boxplot of requirements by year')
plt.show()




num_bins = 5
plt.hist([data_2019, data_2020, data_2021], bins=num_bins, label=['2019', '2020', '2021'])

plt.xlabel('MWh')
plt.ylabel('Frequency')
plt.title('Requirements per year')
plt.legend()
plt.show()

#%% STUDY MAE OF HGB PREDICTION IN FINALPRICES RANGE

confidence = 0.9
price_low = finalPrices.loc[finalPrices.index.year >= 2019]['finalPrice'].quantile((1-confidence)/2)
price_high = finalPrices.loc[finalPrices.index.year >= 2019]['finalPrice'].quantile((1+confidence)/2)
Q_low = max(i for i, val in enumerate(p) if val < price_low)
Q_high = min(i for i, val in enumerate(p) if price_high < val)


Q_high=50
Y_test_b = Y_test.iloc[:,Q_low:Q_high]
Y_test_preds_b = Y_test_preds.iloc[:,Q_low:Q_high]
Y_lag24_b = Y_lag24.iloc[:,Q_low:Q_high]
mae_list_toLag24 = []
mae_list_toRealValue = []
for index in Y_test_preds_b.index:
    y_test_pred_row = Y_test_preds_b.loc[index]
    y_lag24_row = Y_lag24_b.loc[index]
    y_realValue_row = Y_test_b.loc[index]
    mae_lag24 = mean_absolute_error(y_test_pred_row, y_lag24_row)
    mae_realValue = mean_absolute_error(y_test_pred_row, y_realValue_row)
    mae_list_toLag24.append(mae_lag24)
    mae_list_toRealValue.append(mae_realValue)
mae_b_df = pd.DataFrame({'MAE_toLag24': mae_list_toLag24, 'MAE_toRealValue': mae_list_toRealValue}, index=Y_test_preds.index)

color_dict = {2019: 'red', 2020: 'yellow', 2021: 'green'}
scatter = plt.scatter(finalPrices_errors['pred'], mae_b_df['MAE_toRealValue'], alpha=0.5, s=1, c=finalPrices_errors.index.year.map(color_dict))
plt.xlabel('Error in clearing price prediction (€/MWh)')
plt.ylabel('MAE to real curve')
plt.title('Curve prediction and price error')
handles, labels = scatter.legend_elements()
legend_elements = []
for year in finalPrices_errors.index.year.unique():
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=str(year), markerfacecolor=color_dict[year], markersize=5))
plt.legend(handles=legend_elements, title='Year')
plt.savefig('conclusions_scatterplot.png', dpi=300)
plt.show()


