import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import imageio
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
import geopandas as gpd
import imageio
import mapclassify as mc
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']

data = pd.read_csv("dane_out_2016_2020.csv")

explodes = (0, 0.3)
size = data.PM10_Al_Krasińskiego.isna().value_counts()

PM10_Al_Krasińskiego_data = data[['data', 'godzina', 'temperatura_powietrza', 'PM10_Al_Krasińskiego']]
# PM10_Al_Krasińskiego_data['PM10_Al_Krasińskiego']=PM10_Al_Krasińskiego_data.PM10_Al_Krasińskiego.fillna(method='bfill')


PM10_Al_Krasińskiego_data.data = pd.to_datetime(PM10_Al_Krasińskiego_data.data)

YEAR = []
MONTH = []
DAY = []
WEEKDAY = []
for i in range(len(PM10_Al_Krasińskiego_data)):
    WEEKDAY.append(PM10_Al_Krasińskiego_data.data[i].weekday())
    DAY.append(PM10_Al_Krasińskiego_data.data[i].day)
    MONTH.append(PM10_Al_Krasińskiego_data.data[i].month)
    YEAR.append(PM10_Al_Krasińskiego_data.data[i].year)

# PM10_Al_Krasińskiego_data['Year'] = YEAR
# PM10_Al_Krasińskiego_data['Month'] = MONTH
# PM10_Al_Krasińskiego_data['Day'] = DAY
# PM10_Al_Krasińskiego_data['Weekday'] = WEEKDAY
# change_year_index = []
# change_year = []
# year_list = PM10_Al_Krasińskiego_data['Year'].tolist()
# for y in range(0,len(year_list)-1):
#     if year_list[y]!=year_list[y+1]:
#         change_year.append(year_list[y+1])
#         change_year_index.append(y+1)
#
# print(PM10_Al_Krasińskiego_data.loc[change_year_index].head())