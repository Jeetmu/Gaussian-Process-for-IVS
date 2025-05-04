import pandas as pd
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    if T == 0: 
        return max(0, S - K)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T/365) / (sigma*np.sqrt(T/365))
    d2 = d1 - sigma*np.sqrt(T/365)
    price = S*norm.cdf(d1) - K*np.exp(-r*T/365)*norm.cdf(d2)
    return price