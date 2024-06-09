import streamlit as st
import numpy as np 
import plotly.express as px
import datetime
import json
import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from prophet import Prophet
with open('data.json', 'r') as _json:
    data_string = _json.read()

obj = json.loads(data_string)

crypto_names = obj["crypto_names"]
crypto_symbols = obj["crypto_symbols"]

crypto_dict = dict(zip(crypto_names, crypto_symbols))

crypto_selected = st.selectbox(label = 'Selecione sua cripto:', 
                               options = crypto_dict.keys())

crypto_symbol = crypto_dict[crypto_selected]

data_bitcoin = yf.download(crypto_symbol + "-USD")

data_bitcoin.reset_index(inplace=True)
data_bitcoin = data_bitcoin[["Date", 'Adj Close']]
data_bitcoin.rename(columns={"Date": 'ds', 'Adj Close': "y"}, inplace=True)

model = Prophet()
model.fit(data_bitcoin)

period = st.number_input("Insira a quantidade de dias para prever: ", min_value = 0,format = "%i", placeholder = "Type a number")
period = int(period)
data_futuro = model.make_future_dataframe(periods=period)
prever = model.predict(data_futuro)

st.dataframe(prever)
from prophet.plot import plot_components_plotly
from prophet.plot import plot_plotly
plot_components_plotly(model, prever)

st.title(f"Valores de {crypto_selected}")
st.plotly_chart(plot_components_plotly(model, prever))
st.plotly_chart(plot_plotly(model, prever))
