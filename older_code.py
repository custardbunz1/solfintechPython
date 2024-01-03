# Imports
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from prophet import Prophet
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
import yfinance as yf

# LSTM stock predictions on Close
def lstm_prediction(dataframe):
    keras = tf.keras
    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    # Scale down data
    scaler = MinMaxScaler()
    dataframe = scaler.fit_transform(np.array(dataframe).reshape(-1, 1))
    dataframe.shape
    # Use 65% of data for training & 35% for testing
    train_size = int(len(dataframe) * 0.65)
    test_size = len(dataframe) - train_size
    train_data, test_data = dataframe[0:train_size, :], dataframe[train_size:len(dataframe), :1]
    # Create a data matrix
    def create_dataset(dataset, time_step = 1):
        input_data, output_data = [], []
        for i in range(len(dataset)-time_step-1):
           a = dataset[i:(i+time_step), 0]
           input_data.append(a)
           output_data.append(dataset[i + time_step, 0])
        return np.array(input_data), np.array(output_data)

    # calling the create dataset function to split the data into input output datasets with time
    # step 100
    time_step = 100
    input_train, output_train = create_dataset(train_data, time_step)
    input_test, output_test = create_dataset(test_data, time_step)
    # checking values
    print("Checking values:")
    print(input_train.shape)
    print(input_train)
    print(input_test.shape)
    print(output_test.shape)

    # Create and fit LSTM model - 4 layers (1 input, 2 hidden, 1 Dense output) & 50 neurons
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True) # Adam optimizer - mean squared error
    model.summary()

    model.fit(input_train, output_train, validation_data=(input_test, output_test), epochs=2, batch_size=64, verbose=1)

    train_predict = model.predict(input_train)
    test_predict = model.predict(input_test)
    # Transform to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    print("Mean Squared Errors:")
    print(math.sqrt(mean_squared_error(output_train, train_predict)))
    print(math.sqrt(mean_squared_error(output_test, test_predict)))
    # If difference is less than 50 - model is good

    look_back = 100 # Takes the number of values behind the current value
    train_predict_plot = np.empty_like(dataframe)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(dataframe)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataframe) - 1, :] = test_predict
    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataframe))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()


def lstm_prediction2(dataframe):
    keras = tf.keras
    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    BatchNormalization = keras.layers.BatchNormalization

    lstm_dataframe = dataframe.astype('float64')
    lstm_dataframe = lstm_dataframe.fillna(lstm_dataframe.mean())
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array(lstm_dataframe).reshape(-1, 1))
    dataset.shape
    # Split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))

    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        input_data, output_data = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
        input_data.append(a)
        output_data.append(dataset[i + look_back, 0])
        return np.array(input_data), np.array(output_data)

    # Reshape into 'x = t' and 'y = t + 1'
    look_back = 1
    train_input_x, train_output_y = create_dataset(train, look_back)
    test_input_x, test_output_y = create_dataset(test, look_back)
    train_input_x = np.reshape(train_input_x, (train_input_x.shape[0], 1, train_input_x.shape[1]))
    test_input_x = np.reshape(test_input_x, (test_input_x.shape[0], 1, test_input_x.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_input_x, train_output_y, epochs=200, batch_size=1, verbose=2)

    # Make predictions
    train_predict = model.predict(train_input_x)
    test_predict = model.predict(test_input_x)
    # Invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    train_output_y = scaler.inverse_transform([train_output_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_output_y = scaler.inverse_transform([test_output_y])
    # Calculate root mean squared error
    train_score = np.sqrt(mean_squared_error(train_output_y[0], train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (train_score))
    test_score = np.sqrt(mean_squared_error(test_output_y[0], test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_score))

    # Shift train predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict
    print(train_predict_plot)
    print(test_predict_plot)
    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()


def lstm_prediction3(dataframe, chosen_stock):
    keras = tf.keras
    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    dataframe = dataframe.reset_index()[chosen_stock]
    # Scale down data
    scaler = MinMaxScaler()
    dataframe = scaler.fit_transform(np.array(dataframe).reshape(-1, 1))
    train_size = int(len(dataframe) * 0.65)
    test_size = len(dataframe) - train_size
    train_data, test_data = dataframe[0:train_size, :], dataframe[train_size:len(dataframe), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    # calling the create dataset function to split the data into
    # input output datasets with time step 100
    time_step = 10
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # checking values
    print("Checking values:")
    print("Shape of input_train:", X_train.shape)
    print("input_train:", X_train)
    print("Shape of input_test:", X_test.shape)
    print("Shape of output_test:", Y_test.shape)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    y_pred = model.predict(X_test)

    plt.plot(Y_test, marker='.', label="true")
    plt.plot(y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.show()


def get_line_graph(dataframe):
    axis = dataframe.plot.line(y='CTSH')
    plt.show()


def df_plot(data, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()