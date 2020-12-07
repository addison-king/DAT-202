# Final Project - Timeseries Analysis Comparisons
> Brandyn Gilbert
> 
> DAT-202 :: Data Analytics 2

___
# Data Information
Source: https://dataminer2.pjm.com/feed/wind_gen
This site provides the hourly wind generation amounts from all areas owned by PJM. I selected to use only the "Area" RFC (ReliabilityFirst Corporation).
![enter image description here](https://rfirst.org/about/PublishingImages/About%20Us%20Map.png)

The dataset was created by selecting 1-year increments from 2011-2019. The data for 2020 was split in two: January - September and October - November. The first part of 2020 was used in conjunction with the older data. The Oct/Nov data is used for a comparison to our predicted values.
# Data Cleaning (OpenRefine)

Beginning with 10 .csv files containing hourly data over the past 10 years, they were all added into OpenRefine. 
* First was to check for any blank cells (none were found.)
* Removed UTC column
* Removed "area" column
* Removed "File" column
* Alter EST column (cell >> transform >> "toDate(value, "M/d/y hh:mm:ss aa")")
* Sorted by EST time column.
	* "Reorder rows permanently"
* Prepare for prophet:
	* replace(value, "T"," ")
	* replace(value, "Z","")

# Part I: Prophet
* Using the quickstart example from: https://facebook.github.io/prophet/docs/quick_start.html#python-api
* And using the in-class example from Coral:
	*	featuring "DUQ_hourly.csv"
---
We start by importing our .csv that will be used for training and renaming the columns as required by Prophet:

    df_past = pd.read_csv('Past_Data.csv')
    df_past = df_past.rename(columns={"datetime_beginning_ept": "ds", "wind_generation_mw": "y"})

The datetime column isn't in datetime64 format. Let's fix it.

    df_past['ds']=pd.to_datetime(df_past['ds'])
Time to create our model based on our data.

    prophet = Prophet()
    prophet.fit(df_past)

Let's now import our testing data. Rename the columns needed. Change the datetime data type.

    df_future = pd.read_csv('Actual_data.csv')
    df_future = df_future.rename(columns={"datetime_beginning_ept": "ds", "wind_generation_mw": "y"})
    df_future['ds']=pd.to_datetime(df_future['ds'])

Using prophet, we can create a prediction.

    forecast = prophet.predict(df=df_future)
Finally we can see how we did. Plot the prediction vs the historical data. Calculate the root mean square error. Plot the predicted data vs the actual data.

    forecast_plot = prophet.plot(forecast)
    
    prophet_RMSE_value = sqrt(mean_squared_error(df_future['y'], forecast['yhat']))

> Root mean squared error (RMSE) = 2,168

Plot the predicted data vs the actual data.

    plot = pyplot.figure(figsize=(14,8))
    
    ax1 = plot.add_subplot(311)
    ax1.plot_date(x=df_future["ds"], y=df_future["y"], fmt="k-")
    ax1.plot_date(x=forecast["ds"], y=forecast["yhat"], fmt="g-")
    ax1.set_ylim(0,9000)
    pyplot.ylabel('MW Generation')
    pyplot.title('Wind Generation - Prophet')
    #
    ax2 = plot.add_subplot(312)
    ax2.plot_date(x=df_future["ds"], y=df_future["y"], fmt="k-")
    ax2.plot_date(x=forecast["ds"], y=forecast["yhat"], fmt="g-")
    ax2.set_ylim(0,9000)
    ax2.set_xlim([datetime.date(2020,10,1), datetime.date(2020,10,11)])
    pyplot.ylabel('MW Generation')
    
    #
    ax3 = plot.add_subplot(313)
    ax3.plot_date(x=df_future["ds"], y=df_future["y"], fmt="k-")
    ax3.plot_date(x=forecast["ds"], y=forecast["yhat"], fmt="g-")
    ax3.set_ylim(0,9000)
    ax3.set_xlim([datetime.date(2020,10,1), datetime.date(2020,10,6)])
    pyplot.xlabel('Datetime point number')
    pyplot.ylabel('MW Generation')

![enter image description here](https://raw.githubusercontent.com/brandyn-gilbert/DAT-202/main/Final%20Project/Prophet_Graph.png)

# Part II: AR (AutoRegressive)
Begin by importing both .csv files (historical data and test data). Then extract the values from each series (values being MW amounts).

    series_past = read_csv('Past_data.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    training = series_past.values
    
    series_current = read_csv('Actual_data.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    test = series_current.values

Next we build our AR model. 

    lags=1000
    model = AutoReg(training, lags=lags, old_names=False)
    model_fit = model.fit()
Finally we use our model to create predictions and compute RMSE values. (Note: I ran this a few times with different lag values to see what would work best/time).

    predictions = model_fit.predict(start=len(training), end=len(training)+len(test)-1, dynamic=False)
    rmse = sqrt(mean_squared_error(test, predictions))
|Lag ##|RMSE  |
|--|--|
| 50|2798  |
| 100 |2764  |
| 200 | 2660 |
| 500 | 2580 |
| 1000 |2572  |
| 2000 | 2354 |
|5000  |2342  |


Let's visualize how close our model is.

    plot = pyplot.figure(figsize=(14,8))
    
    ax1 = plot.add_subplot(311)
    pyplot.plot(test, color='black')
    pyplot.plot(predictions, color='green')
    pyplot.ylabel('MW Generation')
    pyplot.title('Wind Generation - AR')
    pyplot.gca().legend(('Actual','Predicted'), loc='upper left')
    #
    ax2 = plot.add_subplot(312)
    pyplot.plot(test, color='black')
    pyplot.plot(predictions, color='green')
    ax2.set_xlim([0, 240])
    pyplot.ylabel('MW Generation')
    pyplot.gca().legend(('Actual','Predicted'), loc='upper left')
    #
    ax3 = plot.add_subplot(313)
    pyplot.plot(test, color='black')
    pyplot.plot(predictions, color='green')
    ax3.set_xlim([0, 120])
    pyplot.xlabel('Datetime point number')
    pyplot.ylabel('MW Generation')
    pyplot.gca().legend(('Actual','Predicted'), loc='upper left')

![enter image description here](https://raw.githubusercontent.com/brandyn-gilbert/DAT-202/main/Final%20Project/AR_Graph.png)



# Part III: ARMA (AutoRegressive Moving Average)
Begin the same way as the previous model: import .csv, extract the MW values, and check for any null values.

    series_past = read_csv('Past_data.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    series_current = read_csv('Actual_data.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
    training = DataFrame(series_past.values)
    test = DataFrame(series_current.values)
    
    check = training.isnull().values.any()

We will now create a "lag" dataset to accompany our actual data. Simply, data at t=0 is "lagged" to be t=1. We end up with a dataframe containing two columns: time and time+1.

    df_training_lag = concat([training.shift(1), training], axis=1)
    df_training_lag.columns = ['time', 'time+1']
    
    df_test_lag = concat([test.shift(1), test], axis=1)
    df_test_lag.columns = ['time', 'time+1']
   
    # df_test_lag: position 0, column 'time', is a nAn value. 
    # It needs to be set to the last value in the 'time+1' column from df_training_lag.
    df_test_lag.at[0,'time'] = df_training_lag.at[len(training)-1,'time+1']

Extract the values from each dataframe (creating an array of arrays). Then extract the array of arrays to simply an array of data.

    training_values = df_training_lag.values
    testing_values = df_test_lag.values
    
    training_i, training_actual = training_values[:,0], training_values[:,1]
    testing_i, testing_actual = testing_values[:,0], testing_values[:,1]

Creation of residuals. To do this, we need to convert the array for training (col [0] basically) to a list. Take the resulting list, subtract the "predicted" value from the "actual" value. 

    training_prediction = [x for x in training_i]
    training_residual = []
    for i in range(len(training_prediction)):
        if math.isnan(training_actual[i]-training_prediction[i]) is False:
            training_residual += [training_actual[i]-training_prediction[i]]

Time to create our model. We do this basically the same way as the AR above.

    model = AutoReg(training_residual, lags=1000, old_names=False)
    model_fit = model.fit()
    coef = model_fit.params

With our model, we can now apply our residuals to our predicted data. (For reference, this bit of code was used from the textbook: "Introduction to Time Series Forecasting With Python" by Jason Brownlee).

    span = 20
    history = training_residual[len(training_residual)-span:]
    
    predictions = []
    history_length = len(history)
    
    for i in range(len(testing_actual)):
        yhat = testing_i[i]
        error = testing_actual[i] - yhat
        lag = [history[i] for i in range(history_length-span,history_length)]
        predicted_error = coef[0]
        for j in range(span):
            predicted_error += coef[j+1] * lag[span-j-1]
        yhat = yhat + predicted_error
        predictions.append(yhat)
        history.append(error)

With predicted data, we can: compute RMSE, visually inspect closeness.

    RMSE = sqrt(mean_squared_error(testing_actual, predictions))

> RMSE = 456

    plot = pyplot.figure(figsize=(14,8))
    
    ax1 = plot.add_subplot(311)
    pyplot.plot(testing_actual, color='black')
    pyplot.plot(predictions, color='green')
    pyplot.ylabel('MW Generation')
    pyplot.title('Wind Generation - ARMA')
    pyplot.gca().legend(('Actual','Predicted'), loc='upper left')
    #
    ax2 = plot.add_subplot(312)
    pyplot.plot(testing_actual, color='black')
    pyplot.plot(predictions, color='green')
    ax2.set_xlim([0, 240])
    pyplot.ylabel('MW Generation')
    pyplot.gca().legend(('Actual','Predicted'), loc='upper left')
    #
    ax3 = plot.add_subplot(313)
    pyplot.plot(testing_actual, color='black')
    pyplot.plot(predictions, color='green')
    ax3.set_xlim([0, 120])
    pyplot.xlabel('Datetime point number')
    pyplot.ylabel('MW Generation')
    pyplot.gca().legend(('Actual','Predicted'), loc='upper left')
![enter image description here](https://raw.githubusercontent.com/brandyn-gilbert/DAT-202/main/Final%20Project/MA_Graph.png)



# Part IV: ARIMA (AutoRegressive Integrated Moving Average)

As with models above, read in our .csv files to a series, then extract the MW values.

    series_past = read_csv('Past_data.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    series_current = read_csv('Actual_data.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
    training = series_past.values
    testing = series_current.values

To estimate what lag number we should use, we can look at PACF (partial autocorrelation) graphs. 
![enter image description here](https://raw.githubusercontent.com/brandyn-gilbert/DAT-202/main/Final%20Project/ARIMA_PACF.png)
Easily we can see that using a lag of 3 would be best (3 values are significantly different from 0).
With our lag, we can create our model.

    model = ARIMA(training, order=(3,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()

Now to create a list of historical data from an array. I shrunk the array down from 85,000+ to 20. This was so my laptop would handle the data a little more quickly.

    training_history = [x for x in training]
    training_history = training_history[len(training_history)-20:]

Finally, we will use our model and historical data to create predictions. 

    predictions = []
    for i in range(len(testing)):
        model = ARIMA(training_history, order=(3,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        observation = testing[i]
        training_history.append(observation)

Compute the RMSE.

> RMSE = 299.797

Output the graphics to visually see how close we got.
![enter image description here](https://raw.githubusercontent.com/brandyn-gilbert/DAT-202/main/Final%20Project/ARIMA_Graph.png)


# Part V: Prophet (Try 2)

Building off of the code from the first prophet try, let's see if we can tweak some settings in the model building.
We are going to edit the seasonalities, and changepoint_prior_scale.

    model = Prophet(changepoint_prior_scale=0.80,
                      daily_seasonality=True, 
                      weekly_seasonality=True,
                      yearly_seasonality=True
                     ).add_seasonality(name='hourly',
                                       period=0.04167,
                                       fourier_order=20
                                      )
Sadly this hasn't helped our forecast and our RMSE value is actually worse than before. When looking at the graphs, they look very similar, and not much has changed.
(If anyone has a lot more experience with prophet and can help me with this, I'd appreciate the knowledge).

> RMSE = 2220
![enter image description here](https://raw.githubusercontent.com/brandyn-gilbert/DAT-202/main/Final%20Project/Prophet_2_Graph.png)