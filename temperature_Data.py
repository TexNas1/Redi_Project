import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#Using oop
class TemperatureData:
    def __init__(self):
        self.data = None
        self.top_hot_years = TopHotYears()
    ##fulfilling the requirement of handling file and error
    def TemperatureFileLoad(self, fileName):
        try:
            self.data = pd.read_csv(fileName)
        except FileNotFoundError:
            print("File not found!")
        except pd.errors.EmptyDataError:
            print("Empty data file!")
        except pd.errors.ParserError:
            print("Error parsing the data file!")

    def filter_data(self, start_year=None, end_year=None):
        if start_year and end_year:
            self.data = self.data[(self.data[' Year'] >= start_year) & (self.data[' Year'] <= end_year)]

    def plotChart(self):
        if self.data is not None:
            plt.plot(self.data[' Year'], self.data['Temperature - (Celsius)'])
            plt.xlabel('Year')
            plt.ylabel('Temperature (Celsius)')
            plt.title('Temperature Change Over Time')
            plt.show()
        else:
            print("No data to plot!")

    def generate_statistical_summary(self):
        if self.data is not None:
            summary = self.data['Temperature - (Celsius)'].describe()
            print(summary)
        else:
            print("No data available!")
    
    def train_arima_model(self):
        if self.data is not None:
            tscv = TimeSeriesSplit(n_splits=3)
            for train_index, test_index in tscv.split(self.data):
                train_data = self.data.iloc[train_index]
                test_data = self.data.iloc[test_index]
                model = ARIMA(train_data['Temperature - (Celsius)'], order=(1, 0, 0))
                model_fit = model.fit()

                # Forecasting
                forecast = model_fit.predict(start=test_index[0], end=test_index[-1])
                print("Forecast:")
                print(forecast)
                plt.plot(test_data[' Year'], test_data['Temperature - (Celsius)'], label='Actual')
                plt.plot(test_data[' Year'], forecast, label='Forecast')
                plt.xlabel('Year')
                plt.ylabel('Temperature (Celsius)')
                plt.title('ARIMA Forecast')
                plt.legend()
                plt.show()

        else:
            print("No data available!")

    def train_lstm_model(self):
        if self.data is not None:
            # Prepare input and output data
            train_X = self.data['Temperature - (Celsius)'].shift(1).fillna(0).values.reshape(-1, 1)
            train_y = self.data['Temperature - (Celsius)'].values.reshape(-1, 1)

            # Create LSTM model
            model = Sequential()
            model.add(LSTM(128, activation='relu', input_shape=(1, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            # Train the model
            model.fit(train_X, train_y, epochs=10)
            # Generate forecast
            forecast = model.predict(train_X[-10:])
            print("Forecast:")
            print(forecast)

        else:
            print("No data available!")


class TopHotYears:
    def __init__(self):
        self.top_temperatures = pd.DataFrame(columns=[' Year', 'Temperature - (Celsius)'])
    def find_top_hot_years(self, data, n=10):
        if data is not None:
            sorted_df = data.sort_values('Temperature - (Celsius)', ascending=False)
            top_n_years = sorted_df[' Year'].unique()[:n]
            for year in top_n_years:
                max_temperature_year = sorted_df[sorted_df[' Year'] == year].iloc[0]
                self.top_temperatures = pd.concat([self.top_temperatures, max_temperature_year.to_frame().T])
            return self.top_temperatures[[' Year', 'Temperature - (Celsius)']]
        else:
            print("No data available!")

temp_data = TemperatureData()
temp_data.TemperatureFileLoad('Tempreture_1901_2016_Pakistan.csv')
temp_data.filter_data(1910, 2010)
temp_data.plotChart()
temp_data.generate_statistical_summary()
temp_data.train_arima_model()
temp_data.train_lstm_model()
# Find the top 10 hot years
top_hot_years = temp_data.top_hot_years.find_top_hot_years(temp_data.data, 10)
print(top_hot_years)
