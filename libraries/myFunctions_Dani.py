from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler

def PCA_analysis(X, labels=None):
    
    # Plots some results from the PCA of X. Also if labels is provided, plots a scatter plot with the first 2 PCs colored by label
    
    pca = make_pipeline(StandardScaler(),  PCA()).fit(X)
    princ_comp = pca[1].transform(X)
    
    # Plot of correlation matrix
    plt.imshow(X.corr(), cmap='coolwarm')
    plt.colorbar()
    plt.axis('off')
    plt.show()
    
    # Print cumsum explained variance
    explained_var = pca[1].explained_variance_ratio_
    cum_explained_var_pct = np.round(100*np.cumsum(explained_var), 2)
    n_components = len(cum_explained_var_pct)
    for i in range(n_components):
        print(f"{i+1}: {cum_explained_var_pct[i]:.2f}", end=' / ')
    
    
    # Plot of explained variance
    plt.plot(np.cumsum(pca[1].explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    
    def aux_PCA_scattPlotLabels(score,coeff,labels=None):

        y = labels
        classes = np.unique(y)
        colors = plt.cm.Set1(len(classes))
        color_dict = dict(zip(classes, colors))

        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex,ys * scaley, c = [color_dict[c] for c in y])
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()
    
    if labels is not None:
        aux_PCA_scattPlotLabels(princ_comp[:,0:2],np.transpose(pca[1].components_[0:2, :]),labels)


def substitute_with_PCA(df, cols, n_comps, pca_trained, prefix=None):
    # Substitutes in df those cols for the first n_comps using pca_trained
    # prefix is the prefix of the name of the new cols
    
    if n_comps > len(cols):
        raise ValueError("n_comps cannot be greater than the number of columns in cols")
    
    if prefix is None:
        prefix = 'PC_'
    new_names = [prefix + str(i+1) for i in range(n_comps)]
    
    princ_comp = pca_trained.transform(df[cols])
    pca_subset = princ_comp[:, :n_comps]
    for i, col in enumerate(cols[:n_comps]):
        df[col] = pca_subset[:, i]
        df = df.rename(columns={col: new_names[i]})
    for col in cols[n_comps:]:
        del df[col]
    return df

def TS_seasonal_decompose(TS, from_date, to_date, model='additive'):
    # Plots the seasonal decomposition of a time series
    # model can be additive or multiplicative
    
    from datetime import datetime
    import statsmodels
    
    if from_date is None:
        from_date = TS.index.min()
    if to_date is None:
        to_date = TS.index.max()

    # Convert dates to datetime
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, '%Y-%m-%d')
    if isinstance(to_date, str):
        to_date = datetime.strptime(to_date, '%Y-%m-%d')
    
    # Decompose
    res = statsmodels.tsa.seasonal.seasonal_decompose(TS, model=model)

    observed = res.observed[from_date:to_date]
    trend = res.trend[from_date:to_date]
    seasonal = res.seasonal[from_date:to_date]
    residual = res.resid[from_date:to_date]
    
    #plot the complete time series
    fig, axs = plt.subplots(4, figsize=(16,8))
    axs[0].set_title('OBSERVED', fontsize=16)
    axs[0].plot(observed)
    axs[0].grid()
    
    #plot the trend of the time series
    axs[1].set_title('TREND', fontsize=16)
    axs[1].plot(trend)
    axs[1].grid()
    
    #plot the seasonality of the time series. Period=24 daily seasonality | Period=24*7 weekly seasonality.
    axs[2].set_title('SEASONALITY', fontsize=16)
    axs[2].plot(seasonal)
    axs[2].grid()
    
    #plot the noise of the time series
    axs[3].set_title('NOISE', fontsize=16)
    axs[3].plot(residual)
    axs[3].scatter(y=residual, x=residual.index, alpha=0.5)
    axs[3].grid()
    
    plt.show()



def TS_plots(TS, lags_to_show=30, from_date=None, to_date=None):
    # Plots the TS, ACF and PACF
    
    import matplotlib.pyplot as plt 
    import statsmodels.api as sm
    from datetime import datetime
    
    if from_date is None:
        from_date = TS.index.min()
    if to_date is None:
        to_date = TS.index.max()

    # Convert dates to datetime
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, '%Y-%m-%d')
    if isinstance(to_date, str):
        to_date = datetime.strptime(to_date, '%Y-%m-%d')
    
    # Plot ts
    TS.plot()
    plt.xlim(from_date, to_date)
    plt.show()
    
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    sm.graphics.tsa.plot_acf(TS, lags=lags_to_show, ax=ax1, alpha=0.05)
    ax1.set_title('Autocorrelation Function')
    sm.graphics.tsa.plot_pacf(TS, lags=lags_to_show, ax=ax2, alpha=0.05)
    ax2.set_title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
