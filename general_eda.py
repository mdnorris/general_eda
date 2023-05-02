import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from jedi.api.refactoring import inline
from collections import defaultdict
sns.set_style('darkgrid')
# from rfpimp import *
# from rfpimp import plot_corr_heatmap
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
from sklearn import preprocessing
import shap
# import mkl
from jinja2 import escape
from jinja2 import Markup
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fitter import Fitter, get_common_distributions, get_distributions
import xarray
from sklearn.metrics import mean_squared_error
import dtale
import time


def missing_values(data):
    names = [var for var in int_float.columns]
    missing_count = df[names].isnull().sum()
    var_count = np.array(df[names].isnull().sum() * 100 / len(df)).round(2)
    missing = pd.DataFrame(index=names)
    missing["Count Missing"] = missing_count
    missing["Percent Missing"] = var_count
    print(missing)


def unique(data):
    percent_unique = np.array(100 * df.nunique() / len(df.index)).round(2)
    count_unique = df.nunique()
    names = [var for var in df.columns]
    unique_df = pd.DataFrame(index=names)
    unique_df["Count Unique"] = count_unique
    unique_df["Percent Unique"] = percent_unique
    print(unique_df)


counts = defaultdict(int)
def distributions(data):
    for var in int_float:
        col_names = list(int_float.columns.values)
        dist_test = int_float.dropna().to_numpy()
        dist_list = ['gamma', 'expon', 'cauchy', 'norm', 'uniform']
        f = Fitter(dist_test, distributions=dist_list, timeout=60)
        print(f.fit())
        counts[1] += 1
        if counts[2] < 39:
            counts[2] = 39
        else:
            counts[1] = 38
        print(col_names[counts[1] - 1:counts[2]])
        print(f.summary(plot=True))
        print(f.get_best(method='sumsquare_error'))


def explanation(data):
    print(missing_values(data))
    print(unique(data))


df = pd.read_csv('C:/Users/norri/Documents/GitHub/mercury-ds/attribution/cortex_Push.csv')
df.describe()
df.info()

floater = df.select_dtypes(include='float64')
integ = df.select_dtypes(include='int64')
obj = df.select_dtypes(include='object')

# dtale.show(df)

int_float = pd.concat([floater, integ], axis=1)

corr = int_float.corr(method="pearson")
corr.style.background_gradient(cmap="coolwarm")
sns.heatmap(corr)

explanation(int_float)
distributions(int_float)
