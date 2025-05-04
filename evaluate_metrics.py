import pandas as pd
import numpy as np
from scipy.stats import norm


def evaluate(y_true, y_pred):
    '''
    Root Square Mean Error
    '''
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse

def mean_valuation_error(price_true, price_pred):
    '''
    Mean Valaution Error 
    '''
    mve = np.mean(price_pred - price_true)
    mpe = np.mean((price_pred - price_true)/price_true) * 100
    return mve, mpe
