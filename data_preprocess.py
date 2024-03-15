'''
File: data_preprocess.py
Project: China_Mobile_Sales_Forecast
Description:
-----------
input the source data and preprocess it.
-----------
Author: 626
Created Date: 2023-1106
'''


import pandas as pd


class Data:
    def __init__(self, input_path) -> None:
        self.input_path = input_path
        self.main()

    def load_data(self):
        """Returns a pandas dataframe from the train data set in Kaggle's Demand
        Forecasting competition.
        """
        file_path = self.input_path
        return pd.read_csv(file_path)

    def monthly_sales(self, data):
        """Returns a dataframe where each row represents total sales for a given
        month. Columns include 'date' by month and 'sales'.
        """
        monthly_data = data.copy()
        # Drop the day indicator from the date column
        monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-3])
        # Sum sales per month
        monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
        monthly_data.date = pd.to_datetime(monthly_data.date)
        monthly_data.to_csv('train_data\\monthly_data.csv')
        return monthly_data
    
    def get_diff(self, data):
        """Returns the dataframe with a column for sales difference between each
        month. Results in a stationary time series dataframe. Prior EDA revealed
        that the monthly data was not stationary as it had a time-dependent mean.
        """
        data['sales_diff'] = data.sales.diff()
        data = data.dropna()
        data.to_csv('train_data\\stationary_df.csv')
        return data

    def generate_supervised(self, data):
        """Generates a csv file where each row represents a month and columns
        include sales, the dependent variable, and prior sales for each lag. Based
        on EDA, 12 lag features are generated. Data is used for regression modeling.
        Output df:
        month1  sales  lag1  lag2  lag3 ... lag11 lag12
        month2  sales  lag1  lag2  lag3 ... lag11 lag12
        """
        supervised_df = data.copy()
        #create column for each lag
        for i in range(1, 13):
            col_name = 'lag_' + str(i)
            supervised_df[col_name] = supervised_df['sales_diff'].shift(i)
        #drop null values
        supervised_df = supervised_df.dropna().reset_index(drop=True)
        supervised_df.to_csv('train_data\\model_df.csv', index=False)

    def generate_arima_data(self, data):
        """Generates a csv file with a datetime index and a dependent sales column
        for ARIMA modeling.
        """
        dt_data = data.set_index('date').drop('sales', axis=1)
        dt_data.dropna(axis=0)
        dt_data.to_csv('train_data\\arima_df.csv')

    def main(self):
        """Loads data from Kaggle, generates monthly dataframe and performs
        differencing to create stationarity. Exports csv files for regression
        modeling and for Arima modeling.
        """
        sales_data = self.load_data()
        monthly_df = self.monthly_sales(sales_data)
        stationary_df = self.get_diff(monthly_df)
        self.generate_supervised(stationary_df)
        self.generate_arima_data(stationary_df)


if __name__ == '__main__':
    # input_path = 'forecast_data\\strand_test.csv'
    input_path = 'train_data\\train.csv'
    data = Data(input_path)
