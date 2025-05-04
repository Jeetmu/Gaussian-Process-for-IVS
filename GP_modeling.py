import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def one_day_sample(df):
    """ 
    Sample one day of data from the dataframe 
    """
    one_day_data = df[df['Date'] == '2021-09-01']
    return one_day_data

def processing_data(df):
    '''
    1. Taking the log of the implied volatilty
    2. Creating a new dataframe with log of IV, DTE, Moneyness
    '''
    df_new = pd.DataFrame(columns=['log_IV', 'T', 'M'])
    df_new['log_IV'] = np.log(df['C_IV'])
    df_new['T'] = df['DTE']
    df_new['M'] = df['Moneyness']
    return df_new

def train_test_dataset(df):
    X = df[['M', 'T']].values
    y = df[['log_IV']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def squared_exponential_kernel(X_train1, X_train2, l_m, l_T, sigma_f=1.0):
    M1, T1 = X_train1[:, 0], X_train1[:, 1]
    M2, T2 = X_train2[:, 0], X_train2[:, 1]

    M1 = M1.reshape(-1,1)
    T1 = T1.reshape(-1,1)
    M2 = M2.reshape(-1,1)
    T2 = T2.reshape(-1,1)

    sqdist_M = (M1 - M2.T)**2
    sqdist_T = (T1 - T2.T)**2

    kernel = sigma_f**2 * np.exp(-0.5 * (sqdist_M / l_m**2 + sqdist_T / l_T**2))
    return kernel

df = pd.read_csv('Datasets/call_option_processed.csv')
data = one_day_sample(df)
data = processing_data(data)
data.head(-1)
X_train, X_test, y_train, y_test = train_test_dataset(data)
kernel = squared_exponential_kernel(X_train, X_train, 0.1, 0.1)
kernel