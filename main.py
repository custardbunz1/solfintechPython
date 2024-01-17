# Imports
import customtkinter as tk
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
import yfinance as yf


# GUI settings
tk.set_appearance_mode("system")
tk.set_default_color_theme("blue")

# GUI frame
app = tk.CTk()
app.geometry("780x420")
app.title("SOLFINTECH Intelligent Stock Trader")

# UI elements
title = tk.CTkLabel(app, text="Intelligent Stock Trader")
title.pack(padx=10, pady=10)

# Run GUI
app.mainloop()

# Chosen stocks from NASDAQ-100
chosen_stocks = ['CTSH', 'BKNG', 'REGN', 'MSFT']


def get_tickers():
    # Get list of tickers
    tickers = open("data/nasdaq_100_tickers.txt", "r")
    data = tickers.read().splitlines()

    return data


def get_open_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('open.csv'):
        dataframe = pd.read_csv('open.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Open data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Open']
        data.to_csv('open.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        print(complete_data)
        dataframe = pd.DataFrame(complete_data)

    return dataframe

def get_close_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('close.csv'):
        dataframe = pd.read_csv('close.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Close data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Close']
        data.to_csv('close.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        dataframe = pd.DataFrame(complete_data)
    dataframe.drop(['GEHC'], axis=1, inplace=True) # Dropping GEHC because it contains NULL values

    return dataframe

def get_high_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('high.csv'):
        dataframe = pd.read_csv('high.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download High data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['High']
        data.to_csv('high.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        dataframe = pd.DataFrame(complete_data)

    return dataframe


def get_low_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('low.csv'):
        dataframe = pd.read_csv('low.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Low data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Low']
        data.to_csv('low.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        dataframe = pd.DataFrame(complete_data)

    return dataframe


def get_volume_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('volume.csv'):
        dataframe = pd.read_csv('volume.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Volume data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Volume']
        data.to_csv('volume.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        dataframe = pd.DataFrame(complete_data)

    return dataframe

def get_transposed_ctsh_data(transposed_data_frame):
    get_transposed_ctsh_data = transposed_data_frame.iloc[30]
    return get_transposed_ctsh_data

def get_ctsh_data(dataframe):
    get_ctsh_data = dataframe.iloc[:, 30]
    return get_ctsh_data

def get_ctsh_as_df(dataframe):
    get_ctsh_as_df = dataframe.iloc[:, [30]]
    print(get_ctsh_as_df)
    return get_ctsh_as_df

def get_transposed_bkng_data(transposed_data_frame):
    get_transposed_bkng_data = transposed_data_frame.iloc[17]
    return get_transposed_bkng_data

def get_bkng_data(dataframe):
    get_bkng_data = dataframe.iloc[:, 17]
    return get_bkng_data

def get_transposed_regn_data(transposed_data_frame):
    get_transposed_regn_data = transposed_data_frame.iloc[83]
    return get_transposed_regn_data

def get_regn_data(dataframe):
    get_regn_data = dataframe.iloc[:, 83]
    return get_regn_data

def get_transposed_msft_data(transposed_data_frame):
    get_transposed_msft_data = transposed_data_frame.iloc[66]
    return get_transposed_msft_data

def get_msft_data(dataframe):
    get_msft_data = dataframe.iloc[:, 66]
    return get_msft_data


def transpose_dataframe(dataframe):
    transposed_dataframe = dataframe.T
    # transposed_dataframe.to_csv('transposedDataframe.csv')

    return transposed_dataframe


def change_format(dataframe):
    data_frame_long = pd.melt(dataframe, id_vars='Date', var_name='Stock', value_name='Closing Price')

    return data_frame_long


# Get the mean value for each month - there is too much data if i use daily
def get_monthly_data(dataframe):
    monthly_data_frame = dataframe.resample('M').mean()
    # monthly_data_frame.index = monthly_data_frame.index.strftime('%b %y')

    return monthly_data_frame


def standardise_data(dataframe):
    # Standardise the data - set to have mean of 0 and standard deviation of 1
    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(dataframe))

    return scaled_data


def reduce_data(scaled_data):
    # Reducing the data with PCA from 250 to 10
    pca = PCA(n_components=10)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)
    data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

    return data_pca


def apply_clustering(data_pca):
    # Check if the data has already been clustered
    if os.path.exists('clusters.csv'):
        return
    else:
        # Apply k-means clustering algorithm to group stocks data into clusters of 4
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(data_pca)
        transposed_dataframe['cluster'] = kmeans.labels_
        transposed_dataframe.to_csv('clusters.csv')


def find_correlation(data_frame):
    # Find the Pearson correlations matrix
    if os.path.exists('correlations.csv'):
        correlation = pd.read_csv('correlations.csv', index_col=0)
        return correlation
    else:
        correlation = data_frame.corr(method='pearson')
        correlation.to_csv('correlations.csv')
        return correlation


def rank_correlation(correlation):
    # Find the 11 most positively correlated stocks for each of my stocks
    # (1st one is the stock correlated with itself)
    full_positive_corr_ctsh = close_dataframe.corr()['CTSH'].nlargest(11)
    positive_corr_ctsh = full_positive_corr_ctsh.iloc[1:]
    full_positive_corr_bkng = close_dataframe.corr()['BKNG'].nlargest(11)
    positive_corr_bkng = full_positive_corr_bkng.iloc[1:]
    full_positive_corr_regn = close_dataframe.corr()['REGN'].nlargest(11)
    positive_corr_regn = full_positive_corr_regn.iloc[1:]
    all_positive_corr_msft = close_dataframe.corr()['MSFT'].nlargest(11)
    positive_corr_msft = all_positive_corr_msft.iloc[1:]
    # Find the 10 most negatively correlated stocks for each of my stocks
    negative_corr_ctsh = close_dataframe.corr()['CTSH'].nsmallest(10)
    negative_corr_bkng = close_dataframe.corr()['BKNG'].nsmallest(10)
    negative_corr_regn = close_dataframe.corr()['REGN'].nsmallest(10)
    negative_corr_msft = close_dataframe.corr()['MSFT'].nsmallest(10)
    print('Positive Correlation (CTSH): ', positive_corr_ctsh)
    print('Negative Correlation (CTSH): ', negative_corr_ctsh)
    print('Positive Correlation (BKNG): ', positive_corr_bkng)
    print('Negative Correlation (BKNG): ', negative_corr_bkng)
    print('Positive Correlation (REGN): ', positive_corr_regn)
    print('Negative Correlation (REGN): ', negative_corr_regn)
    print('Positive Correlation (MSFT): ', positive_corr_msft)
    print('Negative Correlation (MSFT): ',negative_corr_msft)


def plot_heatmap(correlation):
    # Plot a heatmap from the data
    heatmap = sn.heatmap(data=correlation)
    plt.title('Correlation Heatmap')
    plt.xlabel('Stock', fontsize=12)
    plt.ylabel('Stock', fontsize=12)
    plt.legend()
    current_figure = plt.gcf()
    image = save_image(current_figure)
    image.save('heatmap.png')
    plt.show()
    return heatmap


def save_image(current_figure):
    # Save the plotted graph as an image
    buffer = io.BytesIO()
    current_figure.savefig(buffer)
    buffer.seek(0)
    image = Image.open(buffer)
    return image


# Univariate analysis
def univariate_analysis(dataframe, stock_name, label):
    counts = dataframe.value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(counts.index, counts)
    plt.title(f'Count Plot of {stock_name} {label}')
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.show()


# Kernel density plot - display skewness of data
def kernel_density_plot(dataframe):
    sn.set_style("darkgrid")
    numericalColumns = dataframe.select_dtypes(include=["int64", "float64"]).columns
    # Plot distribution of each numerical feature
    plt.figure(figsize=(10, len(numericalColumns) * 2))
    for idx, feature in enumerate(numericalColumns, 1):
        plt.subplot(len(numericalColumns), 2, idx)
        sn.histplot(dataframe[feature], kde=True)
        plt.title(f"{feature} | Skewness: {round(dataframe[feature].skew(), 2)}")
    plt.show()


def swarm_plot(dataframe):
    plt.figure(figsize=(12, 8))
    sn.swarmplot(x='Date', y='CTSH', data=dataframe, size=8)
    plt.title(f'Stock Prices Swarm Plot')
    plt.show()


def line_plot(stock, stock_name, label):
    plt.plot(stock)
    plt.title(f'Line Plot of {stock_name} {label}')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.legend()
    plt.show()


# Rolling forecast ARIMA prediction
def arima_prediction(stock_data, stock_name):
    train_data, test_data = stock_data[3:int(len(close_dataframe) * 0.5)], stock_data[int(len(close_dataframe) * 0.5):]
    train_arima = train_data
    test_arima = test_data

    history = [x for x in train_arima] # Previous observations
    y = test_arima
    predictions = list()
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast()[0]
    predictions.append(forecast)
    history.append(y[0])

    for i in range(1, len(y)):
        # Predict
        model = ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast()[0]
        # Invert transformed prediction
        predictions.append(forecast)
        observation = y[i]
        history.append(observation)

    # Report performance
    mean_squared = mean_squared_error(y, predictions)
    print('Mean Squared Error: ' + str(mean_squared))
    mean_absolute = mean_absolute_error(y, predictions)
    print('Mean Absolute Error: ' + str(mean_absolute))
    root_mean_squared = math.sqrt(mean_squared_error(y, predictions))
    print('Root Mean Squared Error: ' + str(root_mean_squared))

    plt.figure(figsize=(16, 8))
    plt.plot(stock_data.index[-600:], stock_data.tail(600), color='blue', label='Train Stock Price')
    plt.plot(test_data.index, y, color='green', label='Actual Stock Price')
    plt.plot(test_data.index, predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{stock_name} Stock Price Prediction')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


# LSTM stock predictions on Close
def lstm_prediction(lstm_dataframe, stock_name):
    keras = tf.keras
    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    lstm_dataframe = close_dataframe.reset_index()[stock_name]
    # Scale down data
    scaler = MinMaxScaler()
    lstm_dataframe = scaler.fit_transform(np.array(lstm_dataframe).reshape(-1, 1))
    # Use 65% of data for training & 35% for testing
    train_size = int(len(lstm_dataframe) * 0.65)
    test_size = len(lstm_dataframe) - train_size
    train_data, test_data = lstm_dataframe[0:train_size, :], lstm_dataframe[train_size:len(lstm_dataframe), :1]
    # Create a data matrix
    def create_dataset(dataset, time_step = 1):
        input_data, output_data = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            input_data.append(a)
            output_data.append(dataset[i + time_step, 0])
        return np.array(input_data), np.array(output_data)

    # calling the create dataset function to split the data into input output datasets with time
    # step 15
    time_step = 15
    input_train, output_train = create_dataset(train_data, time_step)
    input_test, output_test = create_dataset(test_data, time_step)

    # checking values
    print("Checking values:")
    print("Shape of input_train:", input_train.shape)
    print("input_train:", input_train)
    print("Shape of input_test:", input_test.shape)
    print("Shape of output_test:", output_test.shape)

    # Create and fit LSTM model - 4 layers (1 input, 2 hidden, 1 Dense output) & 50 neurons
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')  # Adam optimizer - mean squared error
    model.summary()

    model.fit(input_train, output_train, validation_data=(input_test, output_test), epochs=200, batch_size=64, verbose=1)

    train_predict = model.predict(input_train)
    test_predict = model.predict(input_test)
    # Transform to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    print("Mean Squared Errors:")
    print("output_train: ", mean_squared_error(output_train, train_predict))
    print("output_test: ", mean_squared_error(output_test, test_predict))
    print("Mean Absolute Errors:")
    print("output_train: ", mean_absolute_error(output_train, train_predict))
    print("output_test: ", mean_absolute_error(output_test, test_predict))
    print("Root Mean Squared Errors:")
    print("output_train: ", np.sqrt(mean_squared_error(output_train, train_predict)))
    print("output_test: ", np.sqrt(mean_squared_error(output_test, test_predict)))
    # If difference is less than 50 - model is good

    look_back = 15  # Takes the number of values behind the current value
    train_predict_plot = np.empty_like(lstm_dataframe)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(lstm_dataframe)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(lstm_dataframe) - 1, :] = test_predict

    # Plot baseline and predictions
    plt.title(f'{stock_name} LSTM Stock Price Prediction')
    plt.ylabel('Close Price', fontsize=12)
    plt.xlabel('Time Step', fontsize=12)
    plt.plot(scaler.inverse_transform(lstm_dataframe), color='green', label='Actual Stock Price')
    plt.plot(train_predict_plot, color='blue', label='Train Stock Price')
    plt.plot(test_predict_plot, color='red', label='Predicted Stock Price')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def facebook_prophet_prediction(stock, time_period, stock_name):
    dataframe_prophet = pd.DataFrame(stock)   # Make a new Dataframe with the stock column, because we need ds and y columns
    dataframe_prophet = dataframe_prophet.reset_index()
    dataframe_prophet.columns = ['ds', 'y']
    model = Prophet()
    model.fit(dataframe_prophet)
    future = model.make_future_dataframe(periods=time_period)
    future.tail()
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    forecast_graph = model.plot(forecast)
    components_graph = model.plot_components(forecast)
    plt.ylabel('Close Price', fontsize=12)
    plt.xlabel('Time Step', fontsize=12)
    plt.title(f'{stock_name} Prophet Stock Price Prediction')
    plt.show()


def facebook_prophet_setup(stock, stock_name):
    facebook_prophet_prediction(stock, 7, stock_name)
    facebook_prophet_prediction(stock, 14, stock_name)
    facebook_prophet_prediction(stock, 30, stock_name)



def linear_regression_prediction(dataframe, stock):
    # x = dataframe.index  # DateTimeIndex
    x = dataframe.index.values.astype(float)  # DateTimeIndex
    x = np.asanyarray(x)  # Turn DateTimeIndex into NumPy array
    y = dataframe[stock]  # Pandas Series
    y = np.asanyarray(y)  # Turn Series into NumPy array
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, shuffle=False, random_state=0)
    regression = LinearRegression()
    regression.fit(train_x, train_y)
    print('Train x shape: ', train_x.shape)
    print('Test x shape: ', test_x.shape)
    print('Train y shape: ', train_y.shape)
    print('Test y shape: ', test_y.shape)
    print("Regression coefficient: ", regression.coef_)
    print("Regression intercept: ", regression.intercept_)

    # If R² is closer to 1, the more successful linear regression is at explaining the variation of
    # Y values
    # the coefficient of determination R²
    regression_confidence = regression.score(test_x, test_y)
    print("Linear Regression Confidence: ", regression_confidence)

    predicted = regression.predict(test_x)
    predicted.shape

    # Create table of predicted prices vs real prices
    test_y = test_y.flatten()
    predicted = predicted.flatten()
    dataframe_regression = pd.DataFrame({'Actual_Price': test_y, 'Predicted_Price': predicted})

    print('Mean Absolute Error: ', mean_absolute_error(test_y, predicted))
    print('Mean Squared Error:', mean_squared_error(test_y, predicted))
    print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(test_y, predicted)))

    actual_price_mean = dataframe_regression.Actual_Price.mean()  # Get mean value of Actual Price
    predicted_price_mean = dataframe_regression.Predicted_Price.mean()  # Get mean value of Predicted Price
    accuracy = actual_price_mean / predicted_price_mean * 100
    print("Model accuracy: ", accuracy)

    plt.plot(dataframe_regression.Actual_Price, color='green', label='Actual Stock Price')
    plt.plot(dataframe_regression.Predicted_Price, color='red', label='Predicted Stock Price')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.title(f'{stock} Stock Prediction')
    plt.legend(loc='best')
    plt.show()


def polynomial_regression_prediction(dataframe, stock):
    x = dataframe.index.values.astype(float)  # DateTimeIndex
    x = np.asanyarray(x)  # Turn DateTimeIndex into NumPy array
    y = dataframe[stock]  # Pandas Series
    y = np.asanyarray(y)  # Turn Series into NumPy array
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(x)
    poly.fit(x_poly, y)
    regression = LinearRegression()
    regression.fit(x_poly, y)

    predicted = regression.predict(poly.fit_transform(x))

    print("Regression coefficient: ", regression.coef_)
    print("Regression intercept: ", regression.intercept_)

    regression_confidence = regression.score(x_poly, y)
    print("Linear Regression Confidence: ", regression_confidence)

    print('Mean Absolute Error: ', mean_absolute_error(y, predicted))
    print('Mean Squared Error:', mean_squared_error(y, predicted))
    print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y, predicted)))

    # Visualising the Polynomial Regression results
    plt.scatter(x, y, color='green', label='Actual Stock Price')

    plt.plot(x, predicted,
             color='red', label='Predicted Stock Price')
    plt.title('Polynomial Regression')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.legend()
    plt.show()


def simple_moving_averages(dataframe, stock):
    # Calculate mean of rolling 30/100
    dataframe['SMA30'] = dataframe[stock].rolling(30).mean()
    dataframe['SMA100'] = dataframe[stock].rolling(100).mean()

    # Buy/sell signals
    def buy_sell(dataframe):
        signal_buy = []
        signal_sell = []
        position = False

        for i in range(len(dataframe)):
            if dataframe['SMA30'][i] > dataframe['SMA100'][i]:
                if not position:
                    signal_buy.append(dataframe[stock][i])
                    signal_sell.append(np.nan)
                    position = True
                else:
                    signal_buy.append(np.nan)
                    signal_sell.append(np.nan)
            elif dataframe['SMA30'][i] < dataframe['SMA100'][i]:
                if position:
                    signal_buy.append(np.nan)
                    signal_sell.append(dataframe[stock][i])
                    position = False
                else:
                    signal_buy.append(np.nan)
                    signal_sell.append(np.nan)
            else:
                signal_buy.append(np.nan)
                signal_sell.append(np.nan)
        return pd.Series([signal_buy, signal_sell])

    dataframe['Buy_Signal_price'], dataframe['Sell_Signal_price'] = buy_sell(dataframe)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(dataframe[stock], label=stock, linewidth=0.5, color='green', alpha=0.9)
    ax.plot(dataframe['SMA30'], label='SMA 30', alpha=0.85, color='fuchsia')
    ax.plot(dataframe['SMA100'], label='SMA 100', alpha=0.85, color='purple')
    ax.scatter(dataframe.index, dataframe['Buy_Signal_price'], label='Buy', marker='^', color='green', alpha=1)
    ax.scatter(dataframe.index, dataframe['Sell_Signal_price'], label='Sell', marker='v', color='red', alpha=1)
    ax.set_title(stock + " Price History & Buy / Sell Signals", fontsize=10,
                 color='white')
    ax.set_xlabel('5th Dec 2022 - 5th Dec 2023', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


def wilders_smoothing(data, periods):
    start = np.where(~np.isnan(data))[0][0]  # Check if nans present in beginning
    wilder = np.array([np.nan]*len(data))
    wilder[start+periods-1] = data[start:(start+periods)].mean()  # Simple Moving Average
    for i in range(start+periods,len(data)):
        wilder[i] = (wilder[i-1]*(periods-1) + data[i])/periods  # Wilder's Smoothing
    return(wilder)



# Get and sort the data
tickers = get_tickers()

close_dataframe = get_close_data(tickers)
# open_dataframe = get_open_data(tickers)
# high_dataframe = get_high_data(tickers)
# low_dataframe = get_low_data(tickers)
# volume_dataframe = get_volume_data(tickers)

# transposed_dataframe = transpose_dataframe(close_dataframe)
#
monthly_close_data_frame = get_monthly_data(close_dataframe)
# monthly_open_data_frame = get_monthly_data(open_dataframe)
# monthly_high_data_frame = get_monthly_data(high_dataframe)
# monthly_low_data_frame = get_monthly_data(low_dataframe)
# monthly_volume_data_frame = get_monthly_data(volume_dataframe)

# transposed_ctsh_data = get_transposed_ctsh_data(transposed_dataframe)
# transposed_bkng_data = get_transposed_bkng_data(transposed_dataframe)
# transposed_regn_data = get_transposed_regn_data(transposed_dataframe)
# transposed_msft_data = get_transposed_msft_data(transposed_dataframe)

ctsh_close_data = get_ctsh_data(close_dataframe)
bkng_close_data = get_bkng_data(close_dataframe)
regn_close_data = get_regn_data(close_dataframe)
msft_close_data = get_msft_data(close_dataframe)

# ctsh_open_data = get_ctsh_data(open_dataframe)
# bkng_open_data = get_bkng_data(open_dataframe)
# regn_open_data = get_regn_data(open_dataframe)
# msft_open_data = get_msft_data(open_dataframe)
#
# ctsh_high_data = get_ctsh_data(high_dataframe)
# bkng_high_data = get_bkng_data(high_dataframe)
# regn_high_data = get_regn_data(high_dataframe)
# msft_high_data = get_msft_data(high_dataframe)
#
# ctsh_low_data = get_ctsh_data(low_dataframe)
# bkng_low_data = get_bkng_data(low_dataframe)
# regn_low_data = get_regn_data(low_dataframe)
# msft_low_data = get_msft_data(low_dataframe)
#
# ctsh_volume_data = get_ctsh_data(volume_dataframe)
# bkng_volume_data = get_bkng_data(volume_dataframe)
# regn_volume_data = get_regn_data(volume_dataframe)
# msft_volume_data = get_msft_data(volume_dataframe)

# Clustering
# scaled_data = standardise_data(transposed_dataframe)
# data_pca = reduce_data(scaled_data)
# apply_clustering(data_pca)

# Correlation
# correlation = find_correlation(close_dataframe)
# rank_correlation(correlation)
# plot_heatmap(correlation)

# EDA
i = 0
# stock_close_series = [ctsh_close_data, bkng_close_data, regn_close_data, msft_close_data]
# for stock in stock_close_series:
#     line_plot(stock, chosen_stocks[i], 'Closing Prices')  # Must be a Series
#     univariate_analysis(stock, chosen_stocks[i], 'Closing Prices')  # Must be a Series
#     # swarm_plot(monthly_close_data_frame)  # Must be a Dataframe
#     # kernel_density_plot(close_dataframe)  # Must be a Dataframe
#     i = i + 1

# stock_open_series = [ctsh_open_data, bkng_open_data, regn_open_data, msft_open_data]
# for stock in stock_open_series:
#     line_plot(stock, chosen_stocks[i], 'Opening Prices')
#     univariate_analysis(stock, chosen_stocks[i], 'Opening Prices')
#     # swarm_plot(monthly_open_data_frame)
#     # kernel_density_plot(open_dataframe)
#     i = i + 1
#
# stock_high_series = [ctsh_high_data, bkng_high_data, regn_high_data, msft_high_data]
# for stock in stock_high_series:
#     line_plot(stock, chosen_stocks[i], 'Price Highs')
#     univariate_analysis(stock, chosen_stocks[i], 'Price Highs')
#     swarm_plot(monthly_high_data_frame)
#     kernel_density_plot(high_dataframe)
#     i = i + 1
#
#
# stock_low_series = [ctsh_low_data, bkng_low_data, regn_low_data, msft_low_data]
# for stock in stock_low_series:
#     line_plot(stock, chosen_stocks[i], 'Price Lows')
#     univariate_analysis(stock, chosen_stocks[i], 'Price Lows')
#     swarm_plot(monthly_low_data_frame)
#     kernel_density_plot(low_dataframe)
#     i = i + 1
#
# stock_volume_series = [ctsh_volume_data, bkng_volume_data, regn_volume_data, msft_volume_data]
# for stock in stock_volume_series:
#     line_plot(stock, chosen_stocks[i], 'Trading Volume')
#     univariate_analysis(stock, chosen_stocks[i], 'Price Lows')
#     swarm_plot(monthly_volume_data_frame)
#     kernel_density_plot(volume_dataframe)
#     i = i + 1

# # CTSH Stock Prediction
# arima_prediction(ctsh_close_data, chosen_stocks[0])
# lstm_prediction(close_dataframe, chosen_stocks[0])
facebook_prophet_setup(ctsh_close_data, 'CTSH')
# linear_regression_prediction(close_dataframe, chosen_stocks[0])
# polynomial_regression_prediction(close_dataframe, chosen_stocks[0])
#
# # BKNG Stock Prediction
# arima_prediction(bkng_close_data, chosen_stocks[1])
# lstm_prediction(close_dataframe,  chosen_stocks[1])
facebook_prophet_setup(bkng_close_data, 'BKNG')
# linear_regression_prediction(close_dataframe, chosen_stocks[1])
# polynomial_regression_prediction(close_dataframe, chosen_stocks[1])
#
# # REGN Stock Prediction
# arima_prediction(regn_close_data, chosen_stocks[2])
# lstm_prediction(close_dataframe,  chosen_stocks[2])
facebook_prophet_setup(regn_close_data, 'REGN')
# linear_regression_prediction(close_dataframe, chosen_stocks[2])
# polynomial_regression_prediction(close_dataframe, chosen_stocks[2])
#
# # MSFT Stock Prediction
# arima_prediction(msft_close_data, chosen_stocks[3])
# lstm_prediction(close_dataframe,  chosen_stocks[3])
facebook_prophet_setup(msft_close_data, 'MSFT')
# linear_regression_prediction(close_dataframe, chosen_stocks[3])
# polynomial_regression_prediction(close_dataframe, chosen_stocks[3])
#
# # Technical Analysis
# for stock in chosen_stocks:
#     simple_moving_averages(close_dataframe, stock)


# Each ticker will have 250 days - reduce that to 1. (using PCA) - done
# K-clustering on the ticker data 10 to split them into 4 groups on all 100 tickers
# randomly pick one stock from each of the groups (clusters) created, so you have 4 total
# all analysis for the rest of the report is on those 4 stocks
# for each of the 4 stocks, you need the top 10 positive-negative correlations and analyse them
#
# UI:
# -Drop down menu with each stock, click on it and it will show dataAQ



# 0: CTSH (cell 32) (Cognizant Technology Solutions Corp)
# 1: BKNG (cell 19) (Booking Holdings Inc)
# 2: REGN (cell 83) (Regeneron Pharmaceuticals Inc)
# 3: MSFT (cell 68) (Microsoft Corp)

# Relu: replace negative value with 0