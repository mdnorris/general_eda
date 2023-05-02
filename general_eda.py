import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from rfpimp import *
from rfpimp import plot_corr_heatmap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from sklearn.inspection import permutation_importance
import category_encoders as ce
from sklearn import preprocessing
import shap
# import mkl
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fitter import Fitter, get_common_distributions, get_distributions
import xarray
from sklearn.metrics import mean_squared_error
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
import dtale
import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


print(os.getcwd())

df = pd.read_csv('C:/Users/norri/Documents/GitHub/mercury-ds/attribution/cortex_Push.csv')
df.describe()
df.info()


float = df.select_dtypes(include='float64')
int = df.select_dtypes(include='int64')
object = df.select_dtypes(include='object')

dtale.show(df)

int_float = pd.concat([float, int], axis=1)


def missing_values(df):
    names = [var for var in int_float.columns]
    missing_count = df[names].isnull().sum()
    var_count = np.array(df[names].isnull().sum() * 100 / len(df)).round(2)
    missing = pd.DataFrame(index=names)
    missing["Count Missing"] = missing_count
    missing["Percent Missing"] = var_count
    print(missing)


    def unique(df):
        percent_unique = np.array(100 * df.nunique() / len(df.index)).round(2)
        count_unique = df.nunique()
        names = [var for var in df.columns]
        unique_df = pd.DataFrame(index=names)
        unique_df["Count Unique"] = count_unique
        unique_df["Percent Unique"] = percent_unique
        print(unique_df)


corr_temp = int_float
corr_names = corr_temp.columns.tolist()
temp_df = int_float[corr_names]
corr = temp_df.corr(method="pearson").round(2)
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(18, 18))
cmap = sns.diverging_palette(250, 1, as_cmap=True)
corr_plot = sns.heatmap(corr, annot=True, mask=mask, cmap=cmap,
            vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt(corr_plot)


def distributions(df):
    for var in int_float:
        col_names = list(int_float.columns.values)
        dist_test = int_float.dropna().to_numpy()
        dist_list = ['gamma', 'expon', 'cauchy', 'norm', 'uniform']
        f = Fitter(dist_test, distributions=dist_list, timeout=60)
        f.fit()
        # print(var)
        f.summary(plot=True)
        f.get_best(method='sumsquare_error')


    def summary(df):
        print(missing_values(df))
        print(unique(df))


distributions(int_float)
summary(int_float)
print(corr_plot(int_float))
