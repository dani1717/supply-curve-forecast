# Daniel Foronda
# 2023
# Some functions for the Trabajo Fin de Máster in uc3m


#%%

def errors_analysis(Y,Y_pred,xticklabels=False, percentiles=[.25,.5,.75,.9,.95,.99], totals=False):
    # Returns a dataframe 'stats' with statistics about the errors between Y and Y_pred
    # If totals==True returns also the MAE and RMSE for the whole set
    # Also plots a boxplot of errors distribution
    
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import numpy as np
    
    Y = pd.DataFrame(Y)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.columns = Y.columns
    errors = Y_pred - Y
    
    # Statistics of errors
    stats = errors.describe(percentiles = percentiles)
    stats = stats.round(2)
    
    # avg_error by columns and total
    MAE_byCols = pd.DataFrame(errors.abs().mean(axis=0)).T
    MAE = MAE_byCols.mean(axis=1)[0]
    RMSE_byCols = pd.DataFrame(np.sqrt(mean_squared_error(Y, Y_pred, multioutput='raw_values'))).T
    RMSE_byCols.columns = Y.columns
    RMSE = np.sqrt(mean_squared_error(Y, Y_pred))
    RMSE_byCols = RMSE_byCols.rename(index={0: 1})
    stats = pd.concat([MAE_byCols, RMSE_byCols, stats])
    stats = stats.rename(index={0: "MAE", 1: "RMSE"})
    
    # Move count row to the end of stats
    count_row = stats.loc["count"]
    stats = stats.drop("count")
    stats = pd.concat([stats, count_row.to_frame().T])

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='grey', markersize=1, linestyle='none')
    boxprops = dict(color='#BF0000')
    ax.boxplot(errors, flierprops=flierprops, boxprops=boxprops)
    ax.set_title('Error distributions')
    ax.set_ylabel('error')
    plt.show()
    
    if totals:
        return stats, MAE, RMSE
    if not totals:
        return stats

#%%

def plotCurve(Qs, p=None, xlims=None):
    # Plots one curve. Qs is pd.Series or a list    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    title = ''
    if p==None:
        p = pd.read_csv(r'C:\Users\Dani\Documents\Máster Big Data\TFM\Data\p.csv')
        p = p['x'].tolist()
    if isinstance(Qs, pd.Series):
        title = Qs.name
        Qs = Qs.tolist()
        
    plt.figure(dpi=300)
    plt.step([0] + p , [0] + Qs)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
        start_idx = np.searchsorted(p, xlims[0])
        end_idx = np.searchsorted(p, xlims[1])
        Qs_in_xlims = Qs[start_idx:end_idx]
        plt.ylim(min(Qs_in_xlims)*0.99, max(Qs_in_xlims)*1.01)
    plt.title(title)
    plt.xlabel('price')
    plt.ylabel('MWh')
    plt.show()
    
    
    
#%%
   
def plotPreds(Y_test, Y_pred, serie=None, from_date=None, to_date=None, n_days=None, plotLag24=False):
    # Plots the comparison of prediction and real series for a given Q
    
    import random
    import datetime
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    if n_days is None:
        n_days = 7
    if from_date is None:
        from_date = Y_test.index[random.randint(0, len(Y_test)-n_days*24)]  # We sample a random row from Y_test
    else:
        from_date = datetime.datetime.strptime(from_date, "%Y-%m-%d")
    if to_date is None:
            to_date = from_date + datetime.timedelta(days=n_days)
    if serie is None:
        serie = random.choice(list(Y_test.columns))
    
    Y_pred = pd.DataFrame(Y_pred)
    real_values = Y_test.loc[from_date:to_date, serie]
    rows = list(range(Y_test.index.get_loc(from_date), Y_test.index.get_loc(from_date) + len(real_values)))
    col = Y_test.columns.get_loc(serie)
    pred_values = Y_pred.iloc[rows, col]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_values, color='blue', label='Real values', linewidth=2)
    ax.plot(real_values.index, pred_values, color='red', label='Predicted values', linestyle='--')
    if plotLag24:
        lag_values = Y_test.loc[from_date-datetime.timedelta(hours=24):to_date-datetime.timedelta(hours=24), serie]
        lag_values.index = real_values.index
        ax.plot(lag_values.index, lag_values, color='green', label='lag24 estimator', linestyle='--')
    ax.set_ylabel(serie)
    ax.set_title('Prediction for ' + serie)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.legend()
    plt.show()
    
    
#%%

def restore_monotonicity(df, max_iter=15):
    # Removes iteratively the drops in the rows of df (dataframe) to make it rows monotone increasing
    
    import pandas as pd
    import numpy as np
    
    def colDiffs_df(df):
        # Returns a new df in which the columns are: c1, c2-c1, c3-c2, c4-c3...
        # But in percentage. If c2=100 and c3=95 it will return for c3: -0.05
        
        new_df = pd.DataFrame()
        new_df[df.columns[0]] = df[df.columns[0]]  # Mantener la primera columna sin cambios
        
        for i in range(1, df.shape[1]):
            new_col = (df[df.columns[i]] - df[df.columns[i-1]])
            new_df[df.columns[i]] = new_col
        
        return new_df
    
    
    def count_smallerOnRight(row):
        # Counts for each cell how many cells on the right are lower until a higher value appears
        
        count_row = []
        for i, val in enumerate(row):
            count = 0
            for next_val in row[i+1:]:
                if next_val < val:
                    count += 1
                else:
                    break  # Detenerse cuando se encuentra un punto que supera el punto actual
            count_row.append(count)
        return pd.Series(count_row, index=row.index)
    
    # Not to modify the original dataframe
    df = df.copy()
    # Calculate variations
    df_colDiffs = colDiffs_df(df)
    # Find where there is a drop
    df_drops = df_colDiffs.applymap(lambda x: 1 if x < 0 else 0)
    # Count drops
    n_drops = df_drops.values.sum()
    
    iteration = 1
    while (n_drops != 0 and iteration <= max_iter):
        
        # Apply count_smallerOnRight only on rows with drops
        df_count = df.copy()
        df_count.loc[:,:] = 0
        rows_with_drops = df_drops.any(axis=1)
        df_count.loc[rows_with_drops] = df.loc[rows_with_drops].apply(count_smallerOnRight, axis=1)
    
        
        # IF COUNT > 1 -----------------
        # In this case if there is a very high peak higher than several following points, we are removing it (substituting it for the following value)

        # Select points that are higher than several following points
        rows, cols = np.where(df_count > 1)
        # Change value of that point for the following one
        values = df.values
        values[rows, cols] = values[rows, cols + 1]

        # IF COUNT == 1 ----------------
        # In this case since the drop is just one we are removing the drop 

        # Select pointsthat are higher than just the next one
        rows, cols = np.where(df_count == 1)
        # Remove drop with previous value
        values[rows, cols + 1] = values[rows, cols]
          
        # Count remaining drops
        df_colDiffs = colDiffs_df(df)
        df_drops = df_colDiffs.applymap(lambda x: 1 if x < 0 else 0)
        n_drops = df_drops.values.sum()
        
        #tfm.plotCurve(df.iloc[0,:],p,xlims=[50,200])
        print('After',iteration,'iterations there are still',n_drops,'drops')
        iteration += 1
    
    print('Number of drops after correction:', n_drops)
    
    return df


#%% 
  
def compareCurve(Y_test, Y_pred, p=None, date=None, plotLag24=False, showError=True, reqs=None):
    # Plots the real curve for that date and its prediction (and the lag24 if asked)
    # If reqs are provided, plots final price and its approximation
    
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if p==None:
        p = pd.read_csv(r'C:\Users\Dani\Documents\Máster Big Data\TFM\Data\p.csv')
        p = p['x'].tolist()
        
    if date==None:
        start_row  = 24 if plotLag24 else 0
        date = random.choice(Y_test.index[start_row:])
    
    title = str(date)
    Qs_real = Y_test.loc[date].to_list()
    QS_pred = Y_pred.loc[date].to_list()
    
    plt.figure(dpi=300)
    plt.step([0] + p, [0] + Qs_real, color='blue', linewidth=2, label='Real values')
    plt.step([0] + p, [0] + QS_pred, color='red', linewidth=1, linestyle='dashed', label='Predicted')
    
    if plotLag24:
        Qs_lag24 = Y_pred.loc[date - pd.DateOffset(hours=24)].to_list()
        plt.step([0] + p, [0] + Qs_lag24, color='green', linewidth=1, linestyle='dashed', label='lag24')
        
    if showError: 
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(Qs_real, QS_pred)
        rmse = mean_squared_error(Qs_real, QS_pred, squared=False)
        plt.figtext(0.45, 0.2, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}", fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
   
    if reqs is not None:
        req = reqs.loc[date].values
        plt.axhline(y=req, color='black', linewidth=1, linestyle='dashed', label='Requirement')
       
   
    plt.title(title)
    plt.xlabel('price')
    plt.ylabel('MWh')
    plt.legend()
    plt.show()