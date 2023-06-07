# Redi_Project
This project is for Redi School of Munich. It is final Semester project, consisting of  implementation of temperature rise in Pakistan from 1901 to 2016.
**Temperature Data Analysis**
This repository contains code for analyzing temperature data using various statistical and machine learning models. The code is written in Python and utilizes libraries such as pandas, matplotlib, scikit-learn, statsmodels, and TensorFlow.

**Table of Contents**
Installation
Usage
Description
Methods
Example

**Installation**
To run this code, you need to have Python installed on your system along with the following libraries:
pandas
matplotlib
scikit-learn
statsmodels
TensorFlow

**You can install these libraries using pip:**
    pip install pandas matplotlib scikit-learn statsmodels tensorflow

**Usage**
1.Clone this repository to your local machine or download the code as a ZIP file.
2.Install the required dependencies as mentioned in the Installation section.
3.Open a Python interpreter or create a Python script in your preferred IDE.
4.Import the necessary libraries
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import TimeSeriesSplit
        from statsmodels.tsa.arima.model import ARIMA
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
5.Copy the code from the repository into your Python environment.
6.Customize the code as per your requirements, such as providing the file path for temperature data and setting the desired start and end years.
7.Run the code and observe the results.

**Description**
The provided code performs the following operations:

1.Loads temperature data from a CSV file using the TemperatureFileLoad method.
2.Filters the loaded data based on specified start and end years using the filter_data method.
3.Plots a line chart showing the temperature change over time using the plotChart method.
4.Generates a statistical summary of the temperature data using the generate_statistical_summary method.
5.Trains an ARIMA (AutoRegressive Integrated Moving Average) model and generates forecasts using the train_arima_model method.
6.Trains an LSTM (Long Short-Term Memory) model and generates forecasts using the train_lstm_model method.
7.Finds the top N hottest years in the dataset using the TopHotYears class and the find_top_hot_years method.

**Methods**
The TemperatureData class provides the following methods:

1.TemperatureFileLoad(fileName): Loads temperature data from a CSV file.
2.filter_data(start_year=None, end_year=None): Filters the loaded data based on start and end years.
3.plotChart(): Plots a line chart showing the temperature change over time.
4.generate_statistical_summary(): Generates a statistical summary of the temperature data.
5.train_arima_model(): Trains an ARIMA model and generates forecasts.
6.train_lstm_model(): Trains an LSTM model and generates forecasts.
7.The TopHotYears class provides the following methods:
8.find_top_hot_years(data, n=10): Finds the top N hottest years in the dataset
