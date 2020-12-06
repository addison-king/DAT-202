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



# Part II: AR (AutoRegressive)

# Part III: ARMA (AutoRegressive Moving Average)

# Part IV: ARIMA (AutoRegressive Integrated Moving Average)
