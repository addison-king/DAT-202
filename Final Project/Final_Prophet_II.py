# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from math import sqrt
from fbprophet import Prophet
from fbprophet.plot import plot_weekly
from matplotlib import pyplot
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error, mean_absolute_error
import locale
import datetime
import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'


ascii_cup ="""\
    (  )   (   )  )
     ) (   )  (  (
     ( )  (    ) )
     _____________
    <_____________> ___
    |             |/ _ \\
    |               | | |
    |               |_| |
 ___|             |\___/
/    \___________/    \\
\_____________________/
"""

df_past = pd.read_csv('Past_Data.csv')
df_past = df_past.rename(columns={"datetime_beginning_ept": "ds", "wind_generation_mw": "y"})
df_past['ds']=pd.to_datetime(df_past['ds'])

df_future = pd.read_csv('Actual_data.csv')
df_future = df_future.rename(columns={"datetime_beginning_ept": "ds", "wind_generation_mw": "y"})
df_future['ds']=pd.to_datetime(df_future['ds'])

print("This is going to take a while..")
model = Prophet(changepoint_prior_scale=10, 
                n_changepoints=120,
                daily_seasonality=True, 
                weekly_seasonality=True,
                yearly_seasonality=True
                ).add_seasonality(name='hourly',
                                   period=0.04167,
                                   fourier_order=20
                                  )
                                   

print("Go get a cup of tea while you wait..")
print(ascii_cup)
model.fit(df_past)
print("\ndone")

forecast = model.predict(df=df_future)
print("done")

MAE = mean_absolute_error(df_future['y'], forecast['yhat'])
RMSE = sqrt(mean_squared_error(df_future['y'], forecast['yhat']))

print("Prophet - root mean squared error:", RMSE)
print("Prophet - mean absolute error:", MAE)