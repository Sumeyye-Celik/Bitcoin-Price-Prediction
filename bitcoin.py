# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:47:08 2022

@author: Sümeyye
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

cryptocurrencies = ['BTC-USD']

data = yf.download(cryptocurrencies,  start='2020-09-01',
                end='2022-09-10')

df= data.copy()
# df["Date"] = df.index
# df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
# df.reset_index(drop=True, inplace=True)


print(df.head())
print(df.describe().T)
print(df.info())
print(df.isnull().sum().sort_values(ascending=False))

adj_close=df['Adj Close']
print(adj_close.head())


adj_close.plot(figsize=(20,8))

# returns = adj_close.pct_change().dropna(axis=0)
# print(returns.head())

# df.corr()
# X = df["Date"]
# y=df["Adj Close"]

data_log=np.log(df)

fig, ax=plt.subplots(figsize=(9, 4))
data_log['Adj Close'].plot(ax=ax, label='Log')
ax.legend()

from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
    

data_train = data_log[data_log.index < "2022-08-02"]
data_test = data_log.loc["2022-08-02":"2022-09-01"]
data_predict= data_log.loc["2022-08-29" : "2022-09-08"]


print(f"Train günleri : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test günleri  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")
print(f"Tahmin günleri  : {data_predict.index.min()} --- {data_predict.index.max()}  (n={len(data_predict)})")


fig, ax=plt.subplots(figsize=(9, 4))
data_train['Adj Close'].plot(ax=ax, label='train')
data_test['Adj Close'].plot(ax=ax, label='test')
data_predict["Adj Close"].plot(ax=ax, label="tahmin")
ax.legend()


forecaster = RandomForestRegressor(random_state=123)

forecaster.fit(X=data_train,y=data_train['Adj Close'])
print(forecaster)

#print(data_test["Adj Close"])
predictions = forecaster.predict(data_test)
print(predictions)


print(data_test.shape)
print(predictions.shape)
# predictions = predictions.to_frame()
# print(predictions.shape)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(
    y_true= data_test["Adj Close"], 
    y_pred= predictions)

print(f" Random Forest MSE Hata oranı: {mse} ")

r2 = sklearn.metrics.r2_score(
    y_true=data_test["Adj Close"],
    y_pred=predictions)
print(f" Random Forest R2: {r2}")


mae = sklearn.metrics.mean_absolute_error(
    y_true=data_test["Adj Close"],
    y_pred=predictions)
print(f" Random Forest MAE: {mae}")

rms = np.sqrt(mse)
print(f" Random Forest RMSE: {rms}")

predictions = forecaster.predict(data_predict)
print(predictions)

data_exp = np.exp(predictions)
# data_extrain= np.exp(data_train)
# data_extest= np.exp(data_test)
# data_expred=np.exp(data_predict)

# fig, ax=plt.subplots(figsize=(9, 4))
# data_extrain['Adj Close'].plot(ax=ax, label='train')
# data_extest['Adj Close'].plot(ax=ax, label='test')
    
print(data_predict.index)
print(data_exp) 

pred = pd.DataFrame({"Predictions": predictions},
                    index=pd.date_range(start=df.index[-1], 
                    periods=len(data_predict),
                    freq="D"))

exp_pred=np.exp(pred)
print(exp_pred)

fig, ax=plt.subplots(figsize=(9, 4))
exp_pred.plot(ax=ax, label='pred')




# #########################
# #Linear Regression

# lin_model = sklearn.linear_model.LinearRegression()
# lin_model.fit(data_train, y=data_train['Adj Close'])
# LinearRegression()
                                                                                                                                                                                              
# predictions = lin_model.predict(data_test)
# print(predictions)

# mse1 = mean_squared_error(
#     y_true= data_test["Adj Close"], 
#     y_pred= predictions)

# print(f" Linear Regression MSE Hata oranı: {mse1} ")

# r2 = sklearn.metrics.r2_score(y_true=data_test["Adj Close"],y_pred=predictions)
# mae = sklearn.metrics.mean_absolute_error(y_true=data_test["Adj Close"],y_pred=predictions)
# print(f" R2: {r2}")
# print(f" MAE: {mae}")


# X_predictions = lin_model.predict(data_train['2021-12-11','2021-12-16'])












