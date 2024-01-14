import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import webbrowser

from Front import *


st.title(
    'Your Energy Consumption'
)


# User input for EPC rating
epc_rating = st.selectbox('Select EPC Rating:', ['B', 'C', 'D', 'E'])


prediction_interval = st.radio('Select Prediction Interval:', ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'])


# importing the data
epc_c, epc_d, epc_e, epc_b = get_epc_dfs()

rate =0.3

if epc_rating == 'B':
    preds = train_test_arima_b(epc_b)
    if prediction_interval == 'Hourly':
        preds = sum(preds[:1])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Daily':
        preds = sum(preds[:24])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Weekly':
        preds = sum(preds[:168])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Monthly':
        preds = sum(preds[:168]) * 4
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    else:
        preds = sum(preds[:168]) * 35
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')

elif epc_rating == 'C':
    preds = train_test_arima_c(epc_c)
    if prediction_interval == 'Hourly':
        preds = sum(preds[:1])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Daily':
        preds = sum(preds[:24])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Weekly':
        preds = sum(preds[:168])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Monthly':
        preds = sum(preds[:168]) * 4
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    else:
        preds = sum(preds[:168]) * 35
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')

elif epc_rating == 'D':
    preds = train_test_arima_d(epc_d)
    if prediction_interval == 'Hourly':
        preds = sum(preds[:1])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Daily':
        preds = sum(preds[:24])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Weekly':
        preds = sum(preds[:168])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Monthly':
        preds = sum(preds[:168]) * 4
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    else:
        preds = sum(preds[:168]) * 35
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')

else:
    preds = train_test_arima_e(epc_e)
    if prediction_interval == 'Hourly':
        preds = sum(preds[:1])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Daily':
        preds = sum(preds[:24])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Weekly':
        preds = sum(preds[:168])
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    elif prediction_interval == 'Monthly':
        preds = sum(preds[:168]) * 4
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')
    else:
        preds = sum(preds[:168]) * 35
        rate = f'£{round(preds * rate)}'
        st.markdown(f'{round(preds,2)} kWh')
        st.markdown(f'{rate}')



st.image('/Users/stevenyanez/code/jxsaliba/HeatPumpEnergyPredict/FrontEnd/heatpump_one.jpeg')

# st.image('https://i.pinimg.com/564x/1a/9d/4e/1a9d4e26fc7badfbd4b7f892274554c4.jpg')
# st.markdown('''#
#              Access to your home here
#             ''')
if st.button('Weather Forecast'):
    webbrowser.open_new_tab('https://openweathermap.org/api')



