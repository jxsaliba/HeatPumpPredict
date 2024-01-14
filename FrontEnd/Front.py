import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import warnings
import os
from pmdarima import ARIMA
from pmdarima import model_selection
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')


# print(os.getcwd())
# print(os.chdir('FrontEnd'))


# parent_folder = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# root_path = (os.path.join(parent_folder,'HeatPumpEnergyPredict/FrontEnd'))

# print(root_path, os.getcwd())
def get_epc_dfs():
    # # establishing path to raw_data & our pickle file
    # root_path = (os.path.join(parent_folder,'HeatPumpEnergyPredict/FrontEnd'))
    # os.chdir(root_path)

    # loading our pickle file into a variable
    # dict_df_hour = pickle.load(open('dict_df_hour.pkl','rb'))
    dict_df_hour = pickle.load(open('/Users/stevenyanez/code/jxsaliba/HeatPumpEnergyPredict/FrontEnd/dict_df_hour.pkl','rb'))


    # merging our data

    list_id = dict_df_hour.keys()
    epc = pd.read_csv('/Users/stevenyanez/code/jxsaliba/HeatPumpEnergyPredict/FrontEnd/epc.csv')
    epc = epc[epc['property_id'].isin(list_id)]
    epc_rdx= epc[['property_id','current_energy_efficiency']]

    def assign_rating(rating):
        if rating >= 92:
            return 'A'
        elif rating >= 81:
            return 'B'
        elif rating >= 69:
            return 'C'
        elif rating >= 55:
            return 'D'
        elif rating >=39:
            return 'E'
        elif rating >= 21:
            return 'F'
        else:
            return 'G'


    epc_rdx = epc_rdx.groupby('property_id').max()
    epc_rdx['EPC'] = epc_rdx['current_energy_efficiency'].apply(lambda x: assign_rating(x))

    for k,v in dict_df_hour.items():
        dict_df_hour[k] = dict_df_hour[k].merge(epc_rdx, on=['property_id'], how = 'left')

    for k ,v in dict_df_hour.items():
        df = v.copy()
        df = df.reset_index(drop=True)
        df['energy'] = df['energy'].apply(lambda x: None if x > 100 else x)
        df['energy'] = df['energy'].fillna(method='ffill')
        df['energy'] = df['energy'].fillna(method='bfill')

        df['weather_temp_ma168'] = df['weather_temperature'].rolling(168).mean()
        df['energy_ma168'] = df['energy'].rolling(168).mean()
        df = df.fillna(0)
        dict_df_hour[k] = df

    min_length = []
    for k,v in dict_df_hour.items():
        min_length.append(len(v))
    min_length = min(min_length)

    for k,v in dict_df_hour.items():
        dict_df_hour[k] = dict_df_hour[k][:min_length]

    # classifying our properties into EPC ratings

    epc_c = []
    epc_d = []
    epc_e = []
    epc_b = []

    for k, v in dict_df_hour.items():
        if dict_df_hour[k]['EPC'].iloc[0] == 'C':
            epc_c.append(dict_df_hour[k])
        if dict_df_hour[k]['EPC'].iloc[0] == 'D':
            epc_d.append(dict_df_hour[k])
        if dict_df_hour[k]['EPC'].iloc[0] == 'E':
            epc_e.append(dict_df_hour[k])
        if dict_df_hour[k]['EPC'].iloc[0] == 'B':
            epc_b.append(dict_df_hour[k])

    epc_c = pd.concat(epc_c, ignore_index = True)
    epc_d = pd.concat(epc_d, ignore_index = True)
    epc_e = pd.concat(epc_e,ignore_index = True)
    epc_b = pd.concat(epc_b,ignore_index = True)

    return epc_c, epc_d, epc_e, epc_b

# def train_and_test_arima(selected_property, start_point, num_days):

def train_test_arima_b(epc_b):

    start_point = 3000
    num_days = 20
    length_days = 24*num_days
    end_point = start_point + length_days

    # Extract the relevant data for prediction
    df_b = epc_b[epc_b['property_id']==20408]
    y4 = df_b['energy_ma168'][start_point: end_point]
    exog4 = df_b['weather_temp_ma168'][start_point: end_point]

    train4, test4 = model_selection.train_test_split(y4, train_size=0.65)

    # Model
    modl_b = pm.auto_arima(train4,exog=exog4, start_p=0, start_q=0, start_P=0, start_Q=0,
                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                      stepwise=True, suppress_warnings=True,start_d=0, start_D=0,max_d=12, max_D=12,
                      error_action='ignore')

    # Create predictions for the future, evaluate on test
    preds = modl_b.predict(n_periods=test4.shape[0])

    return preds

def train_test_arima_c(epc_c):

    start_point = 3000
    num_days = 20
    length_days = 24*num_days
    end_point = start_point + length_days

    # Extract the relevant data for prediction
    df_c = epc_c[epc_c['property_id']==20240]
    y = df_c['energy_ma168'][start_point: end_point]
    exog = df_c['weather_temp_ma168'][start_point: end_point]

    train, test = model_selection.train_test_split(y, train_size=0.65)

    # Model
    modl_c = pm.auto_arima(train,exog=exog, start_p=0, start_q=0, start_P=0, start_Q=0,
                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                      stepwise=True, suppress_warnings=True,start_d=0, start_D=0,max_d=12, max_D=12,
                      error_action='ignore')

    # Create predictions for the future, evaluate on test
    preds = modl_c.predict(n_periods=test.shape[0])

    return preds


def train_test_arima_d(epc_d):

    start_point = 3000
    num_days = 20
    length_days = 24*num_days
    end_point = start_point + length_days

    # Extract the relevant data for prediction
    df_d = epc_d[epc_d['property_id']==20101]
    y2 = df_d['energy_ma168'][start_point: end_point]
    exog2 = df_d['weather_temp_ma168'][start_point: end_point]

    train2, test2 = model_selection.train_test_split(y2, train_size=0.65)

    # Model
    modl_d = pm.auto_arima(train2,exog=exog2, start_p=0, start_q=0, start_P=0, start_Q=0,
                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                      stepwise=True, suppress_warnings=True,start_d=0, start_D=0,max_d=12, max_D=12,
                      error_action='ignore')

    # Create predictions for the future, evaluate on test
    preds = modl_d.predict(n_periods=test2.shape[0])

    return preds

def train_test_arima_e(epc_e):

    start_point = 3000
    num_days = 20
    length_days = 24*num_days
    end_point = start_point + length_days

    # Extract the relevant data for prediction
    df_e = epc_e[epc_e['property_id']==20130]
    y3 = df_e['energy_ma168'][start_point: end_point]
    exog3 = df_e['weather_temp_ma168'][start_point: end_point]

    train3, test3 = model_selection.train_test_split(y3, train_size=0.65)

    # Model
    modl_e = pm.auto_arima(train3,exog=exog3, start_p=0, start_q=0, start_P=0, start_Q=0,
                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                      stepwise=True, suppress_warnings=True,start_d=0, start_D=0,max_d=12, max_D=12,
                      error_action='ignore')

    # Create predictions for the future, evaluate on test
    preds = modl_e.predict(n_periods=test3.shape[0])

    return preds
