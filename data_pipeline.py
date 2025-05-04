import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def call_option_data(df):
    '''
    This dataset is a combination of three years of SPDR S&P 500 ETF Trust ($SPY) call options end of day quotes ranging from 01-2020 to 12-2022. 
    Each row represents the information associated with one contract's strike price and a given expiration date.
    '''
    columns = ['Underlying Price', 'Expiration', 'DTE', 'C_IV', 'C_LAST', 'C_BID','C_ASK','STRIKE', 'C_COMBINED_SIZE', 'C_VOLUME']
    df_call = df[columns]
    return df_call

def put_option_data(df):
    '''
    This dataset is a combination of three years of SPDR S&P 500 ETF Trust ($SPY) put options end of day quotes ranging from 01-2020 to 12-2022. 
    Each row represents the information associated with one contract's strike price and a given expiration date.
    '''
    columns = ['Underlying Price', 'Expiration', 'DTE', 'P_IV', 'P_LAST', 'P_BID','P_ASK','STRIKE', 'P_COMBINED_SIZE', 'P_VOLUME']
    df_put = df[columns]
    return df_put

# data cleaning
def call_data_clean(df):
    '''
    1. Option with zero trading volumes are dropped
    2. Options with an implied volatility within 0 (noninclusive) and 1 (inclusive) were maintained in our dataset
    3. Moneyness is within 0.7 - 1.3
    4. Time To maturity short Time period (1-20 DTE), Medium (20-365 DTE), Long (>365 DTE)
    '''
    df = df[df['C_VOLUME']>0]
    df = df[(df['C_IV'] > 0) & (df['C_IV'] <= 1)]
    df['Moneyness'] = df['Underlying Price'] / df['STRIKE']
    df = df[(df['Moneyness']>0.7) & (df['Moneyness'] <= 1.3)]
    df = df[df['DTE']>0]
    df['Period'] = np.where(
        (df['DTE'] > 0) & (df['DTE'] < 20), 'Short',
        np.where(
            (df['DTE'] >= 20) & (df['DTE']<365), 'Medium',
            'Long'
        )
    )
    return df

def put_data_clean(df):
    '''
    1. Option with zero trading volumes are dropped
    2. Options with an implied volatility within 0 (noninclusive) and 1 (inclusive) were maintained in our dataset
    3. Moneyness is within 0.7 - 1.3
    4. Time To maturity short Time period (1-20 DTE), Medium (20-365 DTE), Long (>365 DTE)
    '''
    df = df[df['P_VOLUME']>0]
    df = df[(df['P_IV'] > 0) & (df['P_IV'] <= 1)]
    df['Moneyness'] = df['Underlying Price'] / df['STRIKE']
    df = df[(df['Moneyness']>0.7) & (df['Moneyness'] <= 1.3)]
    df = df[df['DTE']>0]
    df['Period'] = np.where(
        (df['DTE'] > 0) & (df['DTE'] < 20), 'Short',
        np.where(
            (df['DTE'] >= 20) & (df['DTE']<365), 'Medium',
            'Long'
        )
    )
    return df
# making a column to get the ITM, OTM, ATM Option
def call_option_chain(df):
    '''
    ATM = Moneyness Range [0.98, 1.02]
    ITM = Moneyness Range [<0.98]
    OTM = Moneyness Range [>1.02]
    '''
    df['Moneyness_type'] = np.where(
        (df['Moneyness'] > 0.98) & (df['Moneyness'] <=1.02), 'ATM',
        np.where(df['Moneyness'] < 0.98, 'ITM', 'OTM'))
    return df
    
def put_option_chain(df):
    '''
    ATM = Moneyness Range [0.98, 1.02]
    ITM = Moneyness Range [>1.02]
    OTM = Moneyness Range [<0.98]
    '''
    df['Moneyness_type'] = np.where(
        (df['Moneyness'] >= 0.98) & (df['Moneyness'] <= 1.02), 'ATM',
        np.where(df['Moneyness'] > 1.02, 'ITM', 'OTM')
    )
    return df

# Multiplying the Bid Ask Size
def multiply_bid_ask_size(df):
    df[['C_BID_SIZE', 'C_ASK_SIZE']] = df['C_SIZE'].str.split(' x ', expand=True)
    df['C_BID_SIZE'] = pd.to_numeric(df['C_BID_SIZE'], errors='coerce')
    df['C_ASK_SIZE'] = pd.to_numeric(df['C_ASK_SIZE'], errors='coerce')
    df['C_COMBINED_SIZE'] = df['C_BID_SIZE'] * df['C_ASK_SIZE']
    df.drop(columns=['C_SIZE', 'C_BID_SIZE', 'C_ASK_SIZE'], inplace=True)

    df[['P_BID_SIZE', 'P_ASK_SIZE']] = df['P_SIZE'].str.split(' x ', expand=True)
    df['P_BID_SIZE'] = pd.to_numeric(df['P_BID_SIZE'], errors='coerce')
    df['P_ASK_SIZE'] = pd.to_numeric(df['P_ASK_SIZE'], errors='coerce')
    df['P_COMBINED_SIZE'] = df['P_BID_SIZE'] * df['P_ASK_SIZE']
    df.drop(columns=['P_SIZE', 'P_BID_SIZE', 'P_ASK_SIZE'], inplace=True)
    return df


def preprocessed_data(df):
    df = multiply_bid_ask_size(df)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
    
    numeric_columns = ['C_IV', 'C_LAST', 'C_BID', 'C_ASK', 'STRIKE', 'P_BID', 'P_ASK', 'P_LAST', 'P_IV', 'C_VOLUME', 'P_VOLUME']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('Date', inplace=True)
    return df

def output(file):
    df = pd.read_csv(file)
    data = preprocessed_data(df)
    call_data = call_option_data(data)
    put_data = put_option_data(data)
    c_clean = call_data_clean(call_data)
    p_clean = put_data_clean(put_data)
    call_option = call_option_chain(c_clean)
    put_option = put_option_chain(p_clean)
    call_option.to_csv('Datasets/call_option_processed.csv')
    put_option.to_csv('Datasets/put_option_processed.csv')
    return call_option, put_option

input_file = 'Datasets/spy_2020_2022.csv'
call_option, put_option = output(input_file)
call_option.head(-1)
put_option.head(-1)

