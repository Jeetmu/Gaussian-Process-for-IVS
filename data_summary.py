import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_summary(df, option_type='call'):
    '''
    Creating the Testing Data Summary as Mention in the Paper
    '''
    if option_type == 'call':
        iv_col = 'C_IV'
        price_col = 'C_LAST'
    else:
        iv_col = 'P_IV'
        price_col = 'P_LAST'
    
    summary = {}
    
    periods = ['Short', 'Medium', 'Long']
    moneyness_types = ['ATM', 'ITM', 'OTM']

    for period in periods:
        period_data = df[df['Period'] == period]
        summary[period] = {}
        for m_type in moneyness_types:
            subset = period_data[period_data['Moneyness_type'] == m_type]
            count = len(subset)
            avg_iv = subset[iv_col].mean()
            avg_price = subset[price_col].mean()
            summary[period][m_type] = {
                'Count': count,
                'Avg_IV': avg_iv,
                'Avg_Price': avg_price
            }
    total = {}
    for m_type in moneyness_types:
        subset = df[df['Moneyness_type'] == m_type]
        count = len(subset)
        avg_iv = subset[iv_col].mean()
        avg_price = subset[price_col].mean()
        total[m_type] = {
            'Count': count,
            'Avg_IV': avg_iv,
            'Avg_Price': avg_price
        }
    summary['Total'] = total

    return summary

call_option_data = pd.read_csv('Datasets/call_option_processed.csv')
put_option_data = pd.read_csv('Datasets/put_option_processed.csv')
call_summary = create_summary(call_option_data, option_type='call')
put_summary = create_summary(put_option_data, option_type='put')

print(call_summary)
print(put_summary)
