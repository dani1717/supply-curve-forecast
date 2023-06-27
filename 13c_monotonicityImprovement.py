import pandas as pd


data1 = {'Date': ['2019-01-01', '2019-05-01', '2019-09-01', '2020-01-01', '2020-05-01', '2020-09-01', '2021-01-01', '2021-05-01', '2021-09-01'],
         'MAE': [131.274, 123.852, 153.29, 163.232, 136.051, 136.543, 136.847, 147.375, 146.006],
         'RMSE': [167.205, 158.05, 195.893, 209.505, 171.039, 174.369, 176.537, 188.898, 189.128]}

df1 = pd.DataFrame(data1)
df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)

# Datos para el segundo DataFrame
data2 = {'Date': ['2019-01-01', '2019-05-01', '2019-09-01', '2020-01-01', '2020-05-01', '2020-09-01', '2021-01-01', '2021-05-01', '2021-09-01'],
         'MAE': [131.191, 123.572, 152.856, 162.738, 135.493, 136.197, 136.886, 147.324, 145.577],
         'RMSE': [167.042, 157.689, 195.331, 208.897, 170.43, 173.893, 176.585, 188.894, 188.652]}

df2 = pd.DataFrame(data2)
df2['Date'] = pd.to_datetime(df2['Date'])
df2.set_index('Date', inplace=True)

df_mae = df1[['MAE']].join(df2[['MAE']], lsuffix='_df1', rsuffix='_df2')
df_rmse = df1[['RMSE']].join(df2[['RMSE']], lsuffix='_df1', rsuffix='_df2')
df_mae['Incremento %'] = (df_mae['MAE_df2'] - df_mae['MAE_df1']) / df_mae['MAE_df1'] * 100
df_rmse['Incremento %'] = (df_rmse['RMSE_df2'] - df_rmse['RMSE_df1']) / df_rmse['RMSE_df1'] * 100
df_mae = df_mae.round(2)
df_rmse = df_rmse.round(2)

from tabulate import tabulate
table_mae = tabulate(df_mae.transpose(), headers='keys', tablefmt='latex_raw')
table_rmse = tabulate(df_rmse.transpose(), headers='keys', tablefmt='latex_raw')
print(df_rmse)
