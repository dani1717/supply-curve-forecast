import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import TFM_functions as tfm

#%% Import data

p = pd.read_csv(r'Data\p.csv')
p = p['x'].tolist()

folder = "Data/Preprocessed_inputs/"
sufix=''
Y_test = pd.read_csv(folder + 'Y_test' + sufix + '.csv')
Y_test['datetime'] = pd.to_datetime(Y_test['datetime'])
Y_test.set_index('datetime', inplace=True)
Y_pred = np.loadtxt("Data/Preds/10_HGB_preds.csv", delimiter=",")
Y_pred = pd.DataFrame(Y_pred)
Y_pred.index = Y_test.index
Y_pred.columns = Y_test.columns

del folder, sufix, Y_test

#%% Calculate drops

#Calculate differences on each date
def colDiffs_df_pct(df):
    # Returns a new df in which the columns are: c1, c2-c1, c3-c2, c4-c3...
    # But in percentage. If c2=100 and c3=95 it will return for c3: -0.05
    
    new_df = pd.DataFrame()
    new_df[df.columns[0]] = df[df.columns[0]]  # Mantener la primera columna sin cambios
    
    for i in range(1, df.shape[1]):
        new_col = (df[df.columns[i]] - df[df.columns[i-1]])/df[df.columns[i-1]]
        new_df[df.columns[i]] = new_col
    
    return new_df

Y_pred_diffs = colDiffs_df_pct(Y_pred)

# Select negative differences
drops = Y_pred_diffs.apply(lambda row: list(row[row < 0]), axis=1)
drops_all = drops.explode().tolist()
drops_all = pd.DataFrame(drops_all).multiply(100)

#%% Statistics

# Size of the drops
drops_all.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(2)

# Plot of drops
plt.hist(drops_all, bins=100, edgecolor='black', range=(-20, 0))
plt.title('Monotonicity breaks')
plt.xlabel('% of drop')
plt.ylabel('Frequency')
plt.show()

# Number of drops per row
drops_n = drops.map(len)
drops_n.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# Plot some curves
row = Y_pred.iloc[random.randint(0, len(Y_pred) - 1)]
tfm.plotCurve(row,p)
