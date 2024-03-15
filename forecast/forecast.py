



import pickle
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# from model.models import load_data, tts
def load_data(file_name):
    """Returns a pandas dataframe from a csv file."""
    return pd.read_csv(file_name)

def tts(data):
    """Splits the data into train and test. Test set consists of the last 12
    months of data.
    """
    data = data.drop(['sales', 'date'], axis=1)
    train, test = data[0:-12].values, data[-12:].values

    return train, test

def scale_data(train_set, test_set):
    """Scales data using MinMaxScaler and separates data into X_train, y_train,
    X_test, and y_test.

    Keyword Arguments:
    -- train_set: dataset used to train the model
    -- test_set: dataset used to test the model
    """

    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)

    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

    return X_train, y_train, X_test, y_test, scaler

def generate_supervised(data):
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

    supervised_df.to_csv('D:\\Desktop\\python_code\\Sales_Forecasting\\data\\model_df.csv', index=False)

# 加载模型
with open('D:\\Desktop\\python_code\\Sales_Forecasting\\model\\XGBoost_model.pkl', 'rb') as file:
    XGBoost_model = pickle.load(file)

real_data = pd.read_csv('D:\\Desktop\\python_code\\Forecast\\test.csv')
real_data.drop(columns=['id'], inplace=True)
real_data['sales'] = 0
real_data.to_csv('D:\\Desktop\\python_code\\Forecast\\real_data.csv', index=False)
sales_data = load_data()
monthly_df = monthly_sales(sales_data)
stationary_df = get_diff(monthly_df)
generate_supervised(stationary_df)
print(real_data)
Y = XGBoost_model.predict()


# 使用XGBoost模型进行预测
# predictions = XGBoost_model.predict(X_train_scaled)

