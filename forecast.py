'''
File: forecast.py
Project: China_Mobile_Sales_Forecast
Description:
-----------
use trained model to forecast the product sales.
-----------
Author: 626
Created Date: 2023-1106
'''


import pickle
import pandas as pd


class Forecast():
    def __init__(self, trained_model_path, forecast_data_path) -> None:
        self.model = None
        self.trained_model_path = trained_model_path 
        self.forecast_data_path = forecast_data_path
        self.load_model()
        print(self.model)


    def load_model(self):
        with open(trained_model_path, 'rb') as file:
            self.model = pickle.load(file)
            
    
    def forecast(self):
        # self.model.predict()
        pass
    
    def characterize(self):
        pass


# 标准化数据
# real_data = pd.read_csv('data\\test.csv')
# real_data.drop(columns=['id'], inplace=True)
# real_data['sales'] = 0
# print(real_data)
# real_data.to_csv('data\\strand_test.csv', index=False)



# Y = XGBoost_model.predict()

if __name__ == "__main__":
    trained_model_path = "model\\XGBoost_model.pkl"
    forecast_data_path = "forecast_data\\test.csv"
    forecastTools = Forecast(trained_model_path, forecast_data_path)

