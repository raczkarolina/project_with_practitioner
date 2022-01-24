import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings


data = pd.read_csv("dane_out_2016_2020.csv", parse_dates=True)
# print(data.head())

explodes = (0, 0.3)
size = data.PM10_Al_Krasińskiego.isna().value_counts()

PM10_Al_Krasinskiego_data = data[['data', 'godzina', 'PM10_Al_Krasińskiego']]
# print(PM10_Al_Krasinskiego_data[PM10_Al_Krasinskiego_data.isna().any(axis=1)])


# PM10_Al_Krasinskiego_data.temperatura_powietrza = PM10_Al_Krasinskiego_data.temperatura_powietrza.fillna(method='bfill')
PM10_Al_Krasinskiego_data.PM10_Al_Krasińskiego = PM10_Al_Krasinskiego_data.PM10_Al_Krasińskiego.fillna(method='bfill')

PM10_Al_Krasinskiego_data.godzina = PM10_Al_Krasinskiego_data.godzina.fillna('0:00:00')
PM10_Al_Krasinskiego_data.data = PM10_Al_Krasinskiego_data.data.fillna(method='ffill')

PM10_Al_Krasinskiego_data.set_index(['data', 'godzina'], inplace=True, append=False, drop=True)

print(PM10_Al_Krasinskiego_data.head())

PM10_Al_Krasinskiego_data['PM10_Al_Krasińskiego'].plot(figsize=(12, 5))
plt.show()


def ad_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression:", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)


ad_test(PM10_Al_Krasinskiego_data['PM10_Al_Krasińskiego'])

# p> 0,05; Dane nie są stacjonarne
# Nie jest to sztywna reguła, ale dane stacjonarne powinny mieć małą wartość p. Większa wartość p może wskazywać na występowanie pewnych trendów (różna średnia) lub sezonowość.

stepwise_fit = auto_arima(PM10_Al_Krasinskiego_data['PM10_Al_Krasińskiego'], trace=True, suppress_warnings=True)

print(PM10_Al_Krasinskiego_data.shape)
train = PM10_Al_Krasinskiego_data.iloc[:-24]
test = PM10_Al_Krasinskiego_data.iloc[-24:]
print(train.shape, test.shape)

warnings.filterwarnings("ignore")
model = sm.tsa.arima.ARIMA(train['PM10_Al_Krasińskiego'], order=(1, 1, 1))
model = model.fit()
model.summary()


start = len(train)
end = len(train)+len(test)-1
pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA Predictions')
pred.index = PM10_Al_Krasinskiego_data.index[start:end+1]
pred.plot(figsize=(12,10), legend=True)
test['PM10_Al_Krasińskiego'].plot(legend=True)
plt.setp(plt.xticks()[1], rotation=30, ha='right')
plt.show()

test['PM10_Al_Krasińskiego'].mean()
rmse = sqrt(mean_squared_error(pred, test['PM10_Al_Krasińskiego']))
print(rmse)


#PREDICTION
model2 = sm.tsa.arima.ARIMA(PM10_Al_Krasinskiego_data['PM10_Al_Krasińskiego'], order=(1, 1, 1))
model2 = model2.fit()
PM10_Al_Krasinskiego_data.tail()

dt_pred = pd.read_csv("dane/gios-pjp-data.csv", parse_dates=True)
dt_pred = dt_pred[['PM10_Al_Krasińskiego']]
print(dt_pred.head())


index_future_dates = pd.date_range(start='2021-10-30', end='2021-11-06', freq='60min')
pred = model2.predict(start=len(PM10_Al_Krasinskiego_data), end=len(PM10_Al_Krasinskiego_data)+168, typ='levels').rename('ARIMA Predictions')
pred.index = index_future_dates
pred.plot(figsize=(12, 5), legend=True)
dt_pred['PM10_Al_Krasińskiego'].plot(legend=True)
plt.setp(plt.xticks()[1], rotation=30, ha='right')
plt.show()

