# Imports
# import customtkinter as tk
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
import streamlit as st
import tensorflow as tf
import yfinance as yf

# Chosen stocks from NASDAQ-100
chosen_stocks = ['CTSH', 'BKNG', 'REGN', 'MSFT']

def get_tickers():
    # Get list of tickers
    tickers = open("data/nasdaq_100_tickers.txt", "r")
    data = tickers.read().splitlines()

    return data


def get_open_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('data/open.csv'):
        dataframe = pd.read_csv('data/open.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Open data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Open']
        data.to_csv('data/open.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        print(complete_data)
        dataframe = pd.DataFrame(complete_data)

    return dataframe

def get_close_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('data/close.csv'):
        dataframe = pd.read_csv('data/close.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Close data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Close']
        data.to_csv('data/close.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        dataframe = pd.DataFrame(complete_data)
    dataframe.drop(['GEHC'], axis=1, inplace=True) # Dropping GEHC because it contains NULL values

    return dataframe

def get_high_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('data/high.csv'):
        dataframe = pd.read_csv('data/high.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download High data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['High']
        data.to_csv('data/high.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        dataframe = pd.DataFrame(complete_data)

    return dataframe


def get_low_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('data/low.csv'):
        dataframe = pd.read_csv('data/low.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Low data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Low']
        data.to_csv('data/low.csv')
        # Convert array to pandas dataframe, remove NaN values
        complete_data = data.dropna()
        dataframe = pd.DataFrame(complete_data)

    return dataframe


def get_volume_data(data):
    # Check if the data has already been downloaded
    if os.path.exists('data/volume.csv'):
        dataframe = pd.read_csv('data/volume.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download Volume data from Yahoo Finance
        data = yf.download(tickers=data, period='1y', interval='1d')['Volume']
        data.to_csv('data/volume.csv')
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
    get_transposed_regn_data = transposed_data_frame.iloc[80]
    return get_transposed_regn_data

def get_regn_data(dataframe):
    get_regn_data = dataframe.iloc[:, 80]
    return get_regn_data

def get_transposed_msft_data(transposed_data_frame):
    get_transposed_msft_data = transposed_data_frame.iloc[65]
    return get_transposed_msft_data

def get_msft_data(dataframe):
    get_msft_data = dataframe.iloc[:, 65]
    return get_msft_data


def transpose_dataframe(dataframe):
    transposed_dataframe = dataframe.T
    # transposed_dataframe.to_csv('transposedDataframe.csv')

    return transposed_dataframe


def change_format(dataframe):
    data_frame_long = pd.melt(dataframe, id_vars='Date', var_name='Stock', value_name='Closing Price')

    return data_frame_long


# Get the mean value for each month - there is too much data if I use daily
def get_monthly_data(dataframe):
    monthly_data_frame = dataframe.resample('M').mean()

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
    if os.path.exists('data/clusters.csv'):
        return
    else:
        # Apply k-means clustering algorithm to group stocks data into clusters of 4
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(data_pca)
        transposed_dataframe['cluster'] = kmeans.labels_
        transposed_dataframe.to_csv('data/clusters.csv')


def find_correlation(dataframe):
    # Find the Pearson correlations matrix
    if os.path.exists('data/correlations.csv'):
        correlation = pd.read_csv('data/correlations.csv', index_col=0)
        return correlation
    else:
        correlation = dataframe.corr(method='pearson')
        correlation.to_csv('data/correlations.csv')
        return correlation


def rank_correlation(correlation, stock_name):
    # Find the 11 most positively correlated stocks for each of my stocks
    # (1st one is the stock correlated with itself)
    full_positive_corr = close_dataframe.corr()[stock_name].nlargest(11)
    positive_corr = full_positive_corr.iloc[1:]
    # Find the 10 most negatively correlated stocks for each of my stocks
    negative_corr = close_dataframe.corr()[stock_name].nsmallest(10)
    st.write('')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write('')
    with col2:
        st.write(f'Positive Correlation ({stock_name}): ', positive_corr)
    with col3:
        st.write(f'Negative Correlation ({stock_name}): ', negative_corr)
    with col4:
        st.write('')
    st.write('')


# Multivariate analysis
def plot_heatmap(correlation):
    # Plot a heatmap from the data
    heatmap = sn.heatmap(data=correlation)
    plt.title('Correlation Heatmap')
    plt.xlabel('Stock', fontsize=12)
    plt.ylabel('Stock', fontsize=12)
    plt.legend()
    current_figure = plt.gcf()
    image = save_image(current_figure)
    image.save('diagrams/heatmap.png')
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
def bar_plot(dataframe, stock_name, label):
    counts = dataframe.value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(counts.index, counts)
    plt.title(f'Count Plot of {stock_name} {label}')
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.ylim(0, max(counts) * 1.1)
    plt.savefig(f'diagrams/{stock_name}/bar_graph_{label}.png')
    plt.show()


# Density plot - display skewness of data
def density_plot(dataframe, label):
    dataframe.plot.density(figsize=(7, 7), linewidth=4)
    plt.title(f'Density Plot of {label}')
    plt.xlabel(label, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.savefig(f'diagrams/density_plot_{label}.png')
    plt.show()


def swarm_plot(dataframe, stock_name):
    plt.figure(figsize=(12, 8))
    sn.swarmplot(x='Date', y=stock_name, data=dataframe, size=8)
    plt.title(f'Stock Prices Swarm Plot')
    current_figure = plt.gcf()
    image = save_image(current_figure)
    image.save(f'diagrams/{stock_name}/swarm_plot_{stock_name}.png')
    plt.show()


def line_plot(stock, stock_name, label):
    plt.plot(stock)
    plt.title(f'Line Plot of {stock_name} {label}')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.legend()
    plt.savefig(f'diagrams/{stock_name}/line_plot_{label}.png')
    plt.show()
    plt.close()


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
    st.write('Mean Squared Error: ' + str(mean_squared))
    mean_absolute = mean_absolute_error(y, predictions)
    st.write('Mean Absolute Error: ' + str(mean_absolute))
    root_mean_squared = math.sqrt(mean_squared_error(y, predictions))
    st.write('Root Mean Squared Error: ' + str(root_mean_squared))

    plt.figure(figsize=(16, 8))
    plt.plot(stock_data.index[-600:], stock_data.tail(600), color='blue', label='Train Stock Price')
    plt.plot(test_data.index, y, color='green', label='Actual Stock Price')
    plt.plot(test_data.index, predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{stock_name} Stock Price Prediction')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'diagrams/{stock_name}/arima.png')
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

    model.fit(input_train, output_train, validation_data=(input_test, output_test), epochs=1000, batch_size=64, verbose=1)

    train_predict = model.predict(input_train)
    test_predict = model.predict(input_test)
    # Transform to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Mean Squared Errors:")
        st.write("output_train: ", mean_squared_error(output_train, train_predict))
        st.write("output_test: ", mean_squared_error(output_test, test_predict))
    with col2:
        st.write("Mean Absolute Errors:")
        st.write("output_train: ", mean_absolute_error(output_train, train_predict))
        st.write("output_test: ", mean_absolute_error(output_test, test_predict))
    with col3:
        st.write("Root Mean Squared Errors:")
        st.write("output_train: ", np.sqrt(mean_squared_error(output_train, train_predict)))
        st.write("output_test: ", np.sqrt(mean_squared_error(output_test, test_predict)))
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
    plt.savefig(f'diagrams/{stock_name}/lstm.png')
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
    plt.savefig(f'diagrams/{stock_name}/prophet_forecast_{time_period}.png')
    components_graph = model.plot_components(forecast)
    plt.savefig(f'diagrams/{stock_name}/prophet_components_{time_period}.png')
    plt.ylabel('Close Price', fontsize=12)
    plt.xlabel('Time Step', fontsize=12)
    plt.title(f'{stock_name} Prophet Stock Price Prediction')
    plt.show()


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

    predicted = regression.predict(test_x)
    predicted.shape

    # Create table of predicted prices vs real prices
    test_y = test_y.flatten()
    predicted = predicted.flatten()
    dataframe_regression = pd.DataFrame({'Actual_Price': test_y, 'Predicted_Price': predicted})

    actual_price_mean = dataframe_regression.Actual_Price.mean()  # Get mean value of Actual Price
    predicted_price_mean = dataframe_regression.Predicted_Price.mean()  # Get mean value of Predicted Price
    accuracy = actual_price_mean / predicted_price_mean * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Regression coefficient: ", regression.coef_)
        st.write("Regression intercept: ", regression.intercept_)
    with col2:
        # If R² is closer to 1, the more successful linear regression is at explaining the variation of
        # Y values
        # the coefficient of determination R²
        regression_confidence = regression.score(test_x, test_y)
        st.write("Linear Regression Confidence: ", regression_confidence)
    with col3:
        st.write('Mean Absolute Error: ', mean_absolute_error(test_y, predicted))
        st.write('Mean Squared Error:', mean_squared_error(test_y, predicted))
        st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(test_y, predicted)))
        st.write("Model accuracy: ", accuracy)

    plt.plot(dataframe_regression.Actual_Price, color='green', label='Actual Stock Price')
    plt.plot(dataframe_regression.Predicted_Price, color='red', label='Predicted Stock Price')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.title(f'{stock} Linear Regression Stock Prediction')
    plt.legend(loc='best')
    plt.savefig(f'diagrams/{stock}/linear.png')
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Regression coefficient: ", regression.coef_)
    with col2:
        st.write("Regression intercept: ", regression.intercept_)
    with col3:
        regression_confidence = regression.score(x_poly, y)
        st.write('Mean Absolute Error: ', mean_absolute_error(y, predicted))
        st.write('Mean Squared Error:', mean_squared_error(y, predicted))
        st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y, predicted)))
        st.write("Linear Regression Confidence: ", regression_confidence)

    # Visualising the Polynomial Regression results
    plt.scatter(x, y, color='green', label='Actual Stock Price')

    plt.plot(x, predicted,
             color='red', label='Predicted Stock Price')
    plt.title(f'{stock} Polynomial Regression Stock Prediction')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.legend()
    plt.savefig(f'diagrams/{stock}/polynomial.png')
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
    plt.savefig(f'diagrams/{stock}/buy_sell_signals.png')
    plt.show()


def page_layout_stock_selected(stock_open_data, stock_close_data, stock_low_data, stock_high_data, stock_volume_data, stock_name):
    welcome_text = st.markdown(
        "<p style='text-align: center;'>Please note that some graphs and features - especially the Machine Learning model predictions - will take some time to load.</p>",
        unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('')
    with col2:
        analysis_button = st.button('Stock Analysis', type="primary", on_click=analysis_button_state)
    with col3:
        prediction_button = st.button('Stock Value Prediction', type="primary", on_click=prediction_button_state)
    with col4:
        signals_button = st.button('Buy & Sell Signals', type="primary", on_click=signals_button_state)
    with col5:
        st.write('')
    if st.session_state.analysis:
        st.write('Stock Analysis Options:')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            clustering_button = st.button('Stock Clustering', type="secondary")
        with col2:
            correlations_button = st.button('Stock Correlations', type="secondary")
        with col3:
            univariate_button = st.button('Univariate Analysis', type="secondary")
        with col4:
            multivariate_button = st.button('Multivariate Analysis', type="secondary")
        # Clustering
        if clustering_button:
            st.write('')
            st.write(
                f"Below is the dataframe of all stocks' clusters. Scroll to the last column in the dataframe to see the assigned clusters.")
            transposed_dataframe = transpose_dataframe(close_dataframe)
            scaled_data = standardise_data(transposed_dataframe)
            data_pca = reduce_data(scaled_data)
            apply_clustering(data_pca)
            clustering_dataframe = pd.read_csv("data/clusters.csv")
            st.write(clustering_dataframe)
        # Correlation
        if correlations_button:
            correlation = find_correlation(close_dataframe)
            rank_correlation(correlation, stock_name)
            heatmap = plot_heatmap(correlation)
            heatmap = heatmap.get_figure()
            st.pyplot(heatmap)
            st.write('')
        # EDA
        if univariate_button:
            st.write('')
            st.write('Line Plots of Stock Performance:')
            col1, col2, col3 = st.columns(3)
            with col1:
                line_plot(stock_open_data, stock_name, 'Opening Prices')
                st.image(f'diagrams/{stock_name}/line_plot_Opening Prices.png', width=375)
            with col2:
                line_plot(stock_close_data, stock_name, 'Closing Prices')
                st.image(f'diagrams/{stock_name}/line_plot_Closing Prices.png', width=375)
            with col3:
                line_plot(stock_low_data, stock_name, 'Price Lows')
                st.image(f'diagrams/{stock_name}/line_plot_Price Lows.png', width=375)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.write('')
            with col2:
                line_plot(stock_high_data, stock_name, 'Price Highs')
                st.image(f'diagrams/{stock_name}/line_plot_Price Highs.png', width=375)
            with col4:
                line_plot(stock_low_data, stock_name, 'Trading Volume')
                st.image(f'diagrams/{stock_name}/line_plot_Trading Volume.png', width=375)
            with col5:
                st.write('')
            st.write('Bar Graphs of Stock Performance:')
            col1, col2, col3 = st.columns(3)
            with col1:
                bar_plot(stock_open_data, stock_name, 'Opening Prices')
                st.image(f'diagrams/{stock_name}/bar_graph_Opening Prices.png', width=400)
            with col2:
                bar_plot(stock_close_data, stock_name, 'Closing Prices')
                st.image(f'diagrams/{stock_name}/bar_graph_Closing Prices.png', width=400)
            with col3:
                bar_plot(stock_low_data, stock_name, 'Price Lows')
                st.image(f'diagrams/{stock_name}/bar_graph_Price Lows.png', width=400)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.write('')
            with col2:
                bar_plot(stock_high_data, stock_name, 'Price Highs')
                st.image(f'diagrams/{stock_name}/bar_graph_Price Highs.png', width=400)
            with col3:
                st.write('')
            with col4:
                bar_plot(stock_volume_data, stock_name, 'Trading Volume')
                st.image(f'diagrams/{stock_name}/bar_graph_Trading Volume.png', width=400)
                col1, col2, col3 = st.columns(3)
            with col5:
                st.write('')
            st.write('')
            st.write('Swarm Plot displaying monthly average stock price:')
            swarm_plot(monthly_open_data_frame, stock_name)
            st.image(f'diagrams/{stock_name}/swarm_plot_{stock_name}.png', width=1000)
        if multivariate_button:
            st.write('')
            st.write('Density Plots of Stock Closing Price Distribution:')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('')
            with col2:
                density_plot(chosen_stocks_close_dataframe, 'Closing Prices')  # Must be a Dataframe
                st.image(f'diagrams/density_plot_Closing Prices.png', width=700)
            with col3:
                st.write('')
            with col4:
                st.write('')
    elif st.session_state.prediction:
        st.write('')
        st.write('Stock Value Prediction Options:')
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            arima_button = st.button('ARIMA', type="secondary")
        with col2:
            lstm_button = st.button('LSTM', type="secondary")
        with col3:
            prophet_button = st.button('Facebook Prophet', type="secondary", on_click=prophet_state)
        with col4:
            linear_button = st.button('Linear Regression', type="secondary")
        with col5:
            polynomial_button = st.button('Polynomial Regression', type="secondary")
        if arima_button:
            st.write('')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write('')
            with col2:
                st.write('')
            with col3:
                arima_prediction(stock_close_data, stock_name)
            st.image(f'diagrams/{stock_name}/arima.png')
        elif lstm_button:
            st.write('')
            lstm_prediction(close_dataframe, stock_name)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('')
            with col2:
                st.image(f'diagrams/{stock_name}/lstm.png', width=700)
            with col3:
                st.write('')
            with col4:
                st.write('')
        elif prophet_button:
            st.write('')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('')
            with col2:
                st.write('')
                st.write('7 days')
                facebook_prophet_prediction(stock_close_data, 7, stock_name)
                time_period = 7
                st.image(f'diagrams/{stock_name}/prophet_forecast_{time_period}.png', width=550)
                st.image(f'diagrams/{stock_name}/prophet_components_{time_period}.png', width=550)
                st.write('')
                st.write('14 days')
                facebook_prophet_prediction(stock_close_data, 14, stock_name)
                time_period = 14
                st.image(f'diagrams/{stock_name}/prophet_forecast_{time_period}.png', width=550)
                st.image(f'diagrams/{stock_name}/prophet_components_{time_period}.png', width=550)
                st.write('')
                st.write('30 days')
                facebook_prophet_prediction(stock_close_data, 30, stock_name)
                time_period = 30
                st.image(f'diagrams/{stock_name}/prophet_forecast_{time_period}.png', width=550)
                st.image(f'diagrams/{stock_name}/prophet_components_{time_period}.png', width=550)
            with col3:
                st.write('')
            with col4:
                st.write('')
        elif linear_button:
            st.write('')
            linear_regression_prediction(close_dataframe, stock_name)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.write('')
            with col2:
                st.image(f'diagrams/{stock_name}/linear.png', width=750)
            with col3:
                st.write('')
            with col4:
                st.write('')
            with col5:
                st.write('')
        elif polynomial_button:
            st.write('')
            polynomial_regression_prediction(close_dataframe, stock_name)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('')
            with col2:
                st.image(f'diagrams/{stock_name}/polynomial.png', width=700)
            with col3:
                st.write('')
            with col4:
                st.write('')
    elif st.session_state.signals:
        # Technical Analysis
        st.write('')
        st.write('Buy and sell signals calculated from Simple Moving Averages (SMA):')
        simple_moving_averages(close_dataframe, stock_name)
        st.image(f'diagrams/{stock_name}/buy_sell_signals.png')




# Get and sort the data
tickers = get_tickers()

close_dataframe = get_close_data(tickers)
open_dataframe = get_open_data(tickers)
high_dataframe = get_high_data(tickers)
low_dataframe = get_low_data(tickers)
volume_dataframe = get_volume_data(tickers)

monthly_close_data_frame = get_monthly_data(close_dataframe)
monthly_open_data_frame = get_monthly_data(open_dataframe)
monthly_high_data_frame = get_monthly_data(high_dataframe)
monthly_low_data_frame = get_monthly_data(low_dataframe)
monthly_volume_data_frame = get_monthly_data(volume_dataframe)

ctsh_close_data = get_ctsh_data(close_dataframe)
bkng_close_data = get_bkng_data(close_dataframe)
regn_close_data = get_regn_data(close_dataframe)
msft_close_data = get_msft_data(close_dataframe)

ctsh_open_data = get_ctsh_data(open_dataframe)
bkng_open_data = get_bkng_data(open_dataframe)
regn_open_data = get_regn_data(open_dataframe)
msft_open_data = get_msft_data(open_dataframe)

ctsh_high_data = get_ctsh_data(high_dataframe)
bkng_high_data = get_bkng_data(high_dataframe)
regn_high_data = get_regn_data(high_dataframe)
msft_high_data = get_msft_data(high_dataframe)

ctsh_low_data = get_ctsh_data(low_dataframe)
bkng_low_data = get_bkng_data(low_dataframe)
regn_low_data = get_regn_data(low_dataframe)
msft_low_data = get_msft_data(low_dataframe)

ctsh_volume_data = get_ctsh_data(volume_dataframe)
bkng_volume_data = get_bkng_data(volume_dataframe)
regn_volume_data = get_regn_data(volume_dataframe)
msft_volume_data = get_msft_data(volume_dataframe)

stock_open_series = [ctsh_open_data, bkng_open_data, regn_open_data, msft_open_data]
stock_close_series = [ctsh_close_data, bkng_close_data, regn_close_data, msft_close_data]
stock_low_series = [ctsh_low_data, bkng_low_data, regn_low_data, msft_low_data]
stock_high_series = [ctsh_high_data, bkng_high_data, regn_high_data, msft_high_data]
stock_volume_series = [ctsh_volume_data, bkng_volume_data, regn_volume_data, msft_volume_data]

chosen_stocks_open_dataframe = pd.DataFrame(stock_open_series).T
chosen_stocks_close_dataframe = pd.DataFrame(stock_close_series).T
chosen_stocks_low_dataframe = pd.DataFrame(stock_low_series).T
chosen_stocks_high_dataframe = pd.DataFrame(stock_high_series).T
chosen_stocks_volume_dataframe = pd.DataFrame(stock_volume_series).T


# transposed_ctsh_data = get_transposed_ctsh_data(transposed_dataframe)
# transposed_bkng_data = get_transposed_bkng_data(transposed_dataframe)
# transposed_regn_data = get_transposed_regn_data(transposed_dataframe)
# transposed_msft_data = get_transposed_msft_data(transposed_dataframe)

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image('designs/solfintech-logo.png')
with col3:
    st.write(' ')
st.markdown(
        "<p style='text-align: center; color: #ED7D31'><strong><i>Intelligent Stock Trader</i></strong></p>",
        unsafe_allow_html=True)

# Create an empty container for dynamic text
welcome_text = st.empty()

st.write('')

option = st.selectbox(
    'Select stock:',
    (chosen_stocks[0], chosen_stocks[1], chosen_stocks[2], chosen_stocks[3], 'Review - All Stocks'),
    index=None,
    placeholder='None selected',
)

st.markdown('')

if ('signals' not in st.session_state):
    st.session_state.signals = False

if ('analysis' not in st.session_state):
    st.session_state.analysis = False

if ('prediction' not in st.session_state):
    st.session_state.prediction = False

if ('prophet' not in st.session_state):
    st.session_state.prophet = False

def analysis_button_state():
    st.session_state.analysis = True
    st.session_state.prediction = False
    st.session_state.signals = False
    st.session_state.prophet = False
def prediction_button_state():
    st.session_state.prediction = True
    st.session_state.analysis = False
    st.session_state.signals = False
    st.session_state.prophet = False

def signals_button_state():
    st.session_state.signals = True
    st.session_state.prediction = False
    st.session_state.analysis = False
    st.session_state.prophet = False

def prophet_state():
    st.session_state.signals = False
    st.session_state.prediction = True
    st.session_state.analysis = False
    st.session_state.prophet = True


# CTSH
if option == chosen_stocks[0]:
    page_layout_stock_selected(ctsh_open_data, ctsh_close_data, ctsh_low_data, ctsh_high_data, ctsh_volume_data, chosen_stocks[0])
# BKNG
elif option == chosen_stocks[1]:
    page_layout_stock_selected(bkng_open_data, bkng_close_data, bkng_low_data, bkng_high_data, bkng_volume_data, chosen_stocks[1])
# REGN
elif option == chosen_stocks[2]:
    page_layout_stock_selected(regn_open_data, regn_close_data, regn_low_data, regn_high_data, regn_volume_data, chosen_stocks[2])
# MSFT
elif option == chosen_stocks[3]:
    page_layout_stock_selected(msft_open_data, msft_close_data, msft_low_data, msft_high_data, msft_volume_data, chosen_stocks[3])
elif option == 'Review - All Stocks':
    welcome_text = st.markdown(
        "<p style='text-align: center;'>Please note that some graphs and features - especially the Machine Learning model predictions - will take some time to load.</p>",
        unsafe_allow_html=True)
    st.write('')
    st.write('All dataframes of all stocks:')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        view_open_dataframe_button = st.button("Opening Prices", type="secondary")
    with col2:
        view_close_dataframe_button = st.button("Closing Prices", type="secondary")
    with col3:
        view_low_dataframe_button = st.button("Price Lows", type="secondary")
    with col4:
        view_high_dataframe_button = st.button("Price Highs", type="secondary")
    with col5:
        view_volume_dataframe_button = st.button("Trading Volumes", type="secondary")
    if view_open_dataframe_button:
        st.write(open_dataframe)
    if view_close_dataframe_button:
        st.write(close_dataframe)
    if view_low_dataframe_button:
        st.write(low_dataframe)
    if view_high_dataframe_button:
        st.write(high_dataframe)
    if view_volume_dataframe_button:
        st.write(volume_dataframe)
else:
    welcome_text.markdown(
        "<p style='text-align: center;'>Welcome to the SOLFINTECH Intelligent Stock Trader. Select a stock to get started.</p>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #ED7D31';>• CTSH = Cognizant Technology Solutions Corporation</p>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #ED7D31';>• BKNG = Booking Holdings Incorporated</p>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #ED7D31';>• REGN = Regeneron Pharmaceuticals Incorporated</p>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #ED7D31';>• MSFT = Microsoft Corporation</p>",
        unsafe_allow_html=True)