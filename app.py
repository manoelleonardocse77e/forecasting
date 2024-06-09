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

#Primeira parte: Puxa as informacoes das criptomoedas do json, cria a selecao de cripto do usuario, adiciona a cripto escolhida a moeda e cria o dicionario de cripto
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
#Segunda Parte: Reseta o index da data criada, substitui pelos dados utilizado pelo prophet e comeca o modelo preditivo
data_bitcoin.reset_index(inplace=True)
data_bitcoin = data_bitcoin[["Date", 'Adj Close']]
data_bitcoin.rename(columns={"Date": 'ds', 'Adj Close': "y"}, inplace=True)

model = Prophet()
model.fit(data_bitcoin)
#Terceira Parte: Cria a selecao de quantidade de dias que o usuario deseja prever (quanto mais tempo no futuro mais impreciso o algoritmo se torna), com esse periodo o modelo preditivo comeca a agir e prever o valor da criptomoeda selecionada
period = st.number_input("Insira a quantidade de dias para prever: ", min_value = 0,format = "%i", placeholder = "Type a number")
period = int(period)
data_futuro = model.make_future_dataframe(periods=period)
prever = model.predict(data_futuro)

st.dataframe(prever)

#Quarta Parte: importamos a plotagem do prophet para disponibilizarmos os graficos, o primeiro sendo dividido em 3 categorias (mensal anual e semanal) o segundo possuindo um acompanhamento do primeiro ano contido no dataframe ate o dia escolhido para ser previsto
from prophet.plot import plot_components_plotly
from prophet.plot import plot_plotly
plot_components_plotly(model, prever)

st.title(f"Valores de {crypto_selected}")
st.plotly_chart(plot_components_plotly(model, prever))
st.plotly_chart(plot_plotly(model, prever))
