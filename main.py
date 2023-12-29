import os

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt
import io
from PIL import Image
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

chosenStocks = ['CTSH', 'BKNG', 'REGN', 'MSFT']

def getData():
    # Get list of tickers
    tickers = open("dataset/nasdaq_100_tickers.txt", "r")
    data = tickers.read().splitlines()

    # Check if the data has already been downloaded
    if os.path.exists('dataframe.csv'):
        dataFrame = pd.read_csv('dataframe.csv', index_col="Date", parse_dates=True).dropna()
    else:
        # Download data from yfinance
        data = yf.download(tickers=data, period='1y', interval='1d')['Close']
        data.to_csv('dataframe.csv')
        # Remove empty rows and convert array to pandas data frame
        completeData = data.dropna()
        dataFrame = pd.DataFrame(completeData)
    dataFrame.reset_index()

    return dataFrame

def getCtshData(transposedDataFrame):
    getCtshData = transposedDataFrame.iloc[30]
    return getCtshData

def getBkngData(transposedDataFrame):
    getBkngData = transposedDataFrame.iloc[17]
    # getBkngData.to_csv('bkng.csv')
    return getBkngData

def getRegnData(transposedDataFrame):
    getRegnData = transposedDataFrame.iloc[83]
    # getRegnData.to_csv('regn.csv')
    return getRegnData

def getMsftData(transposedDataFrame):
    getMsftData = transposedDataFrame.iloc[66]
    # getMsftData.to_csv('msft.csv')
    return getMsftData

def transposeDataFrame(dataFrame):
    transposedDataFrame = dataFrame.T
    # transposedDataFrame.to_csv('transposedDataframe.csv')

    return transposedDataFrame


def changeFormat(dataFrame):
    dataFrameLong = pd.melt(dataFrame, id_vars='Date', var_name='Stock', value_name='Closing Price')

    return dataFrameLong


def standardizeData(dataFrame):
    # Standardize the data - set to have mean of 0 and standard deviation of 1
    scalar = StandardScaler()
    scaledData = pd.DataFrame(scalar.fit_transform(dataFrame))

    return scaledData


def reduceData(scaledData):
    # Reducing the data with PCA from 250 to 10
    pca = PCA(n_components=10)
    pca.fit(scaledData)
    dataPca = pca.transform(scaledData)
    dataPca = pd.DataFrame(dataPca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

    return dataPca


def applyClustering(dataPca):
    # Check if the data has already been clustered
    if os.path.exists('clusters.csv'):
        return
    else:
        # Apply k-means clustering algorithm to group stocks data into clusters of 4
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(dataPca)
        transposedDataFrame['cluster'] = kmeans.labels_
        transposedDataFrame.to_csv('clusters.csv')


def findCorrelation(dataFrame):
    # Find the Pearson correlations matrix
    if os.path.exists('correlations.csv'):
        correlation = pd.read_csv('correlations.csv', index_col=0)
        return correlation
    else:
        correlation = dataFrame.corr(method='pearson')
        correlation.to_csv('correlations.csv')
        return correlation


def rankCorrelation(correlation):
    # Find the 11 most positively correlated stocks for each of my stocks
    # (1st one is the stock correlated with itself)
    full_positive_corr_ctsh = dataFrame.corr()['CTSH'].nlargest(11)
    positive_corr_ctsh = full_positive_corr_ctsh.iloc[1:]
    full_positive_corr_bkng = dataFrame.corr()['BKNG'].nlargest(11)
    positive_corr_bkng = full_positive_corr_bkng.iloc[1:]
    full_positive_corr_regn = dataFrame.corr()['REGN'].nlargest(11)
    positive_corr_regn = full_positive_corr_regn.iloc[1:]
    all_positive_corr_msft = dataFrame.corr()['MSFT'].nlargest(11)
    positive_corr_msft = all_positive_corr_msft.iloc[1:]
    # Find the 10 most negatively correlated stocks for each of my stocks
    negative_corr_ctsh = dataFrame.corr()['CTSH'].nsmallest(10)
    negative_corr_bkng = dataFrame.corr()['BKNG'].nsmallest(10)
    negative_corr_regn = dataFrame.corr()['REGN'].nsmallest(10)
    negative_corr_msft = dataFrame.corr()['MSFT'].nsmallest(10)
    print(positive_corr_ctsh)
    print(negative_corr_ctsh)
    print(positive_corr_bkng)
    print(negative_corr_bkng)
    print(positive_corr_regn)
    print(negative_corr_regn)
    print(positive_corr_msft)
    print(negative_corr_msft)


def plotHeatmap(correlation):
    # Plot a heatmap from the data
    heatmap = sn.heatmap(data=correlation)
    plt.show()
    currentFigure = plt.gcf()
    image = saveImage(currentFigure)
    image.save('heatmap.png')
    return heatmap


def saveImage(currentFigure):
    # Save the plotted graph as an image
    buffer = io.BytesIO()
    currentFigure.savefig(buffer)
    buffer.seek(0)
    image = Image.open(buffer)
    return image


# Univariate analysis
def univariateAnalysis(dataFrame):
    counts = dataFrame.value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(counts.index, counts)
    plt.title('Count Plot of Stock Closes')
    plt.xlabel('Closes')
    plt.ylabel('Count')
    plt.show()


# Kernel density plot - display skewness of data
def kernelDensityPlot(dataFrame):
    sn.set_style("darkgrid")
    numericalColumns = dataFrame.select_dtypes(include=["int64", "float64"]).columns
    # Plot distribution of each numerical feature
    plt.figure(figsize=(10, len(numericalColumns) * 2))
    for idx, feature in enumerate(numericalColumns, 1):
        plt.subplot(len(numericalColumns), 2, idx)
        sn.histplot(dataFrame[feature], kde=True)
        plt.title(f"{feature} | Skewness: {round(dataFrame[feature].skew(), 2)}")
    plt.show()


def swarmPlot(dataFrame):
    plt.figure(figsize=(12, 8))
    sn.swarmplot(x='Date', y='CTSH', data=dataFrame, size=8)
    plt.title('CTSH Stock Prices Swarm Plot')
    plt.show()


def linePlot(stock, dataFrame):
    dataFrame[[stock]].plot(subplots=True, layout=(4, 1));
    plt.show()


# Get the mean value for each month - there is too much data if i use daily
def getMonthlyData(dataFrame):
    monthlyDataFrame = dataFrame.resample('M').mean()
    # monthlyDataFrame.index = monthlyDataFrame.index.strftime('%b %y')

    return monthlyDataFrame

def arimaPrediction():
    train_data, test_data = dataFrame[0:int(len(dataFrame) * 0.9)], dataFrame[int(len(dataFrame) * 0.9):]
    train_arima = train_data['Open']
    test_arima = test_data['Open']

    history = [x for x in train_arima]
    y = test_arima
    predictions = list()
    model = sm.tsa.arima.ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(y[0])

    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)

# Unused functions
# def pltPlotCtsh(getCtshData):
#     plt.plot(getCtshData)
#     plt.show()
#
#
# def getLineGraph(dataFrame):
#     axis = dataFrame.plot.line(y='CTSH')
#     plt.show()



dataFrame = getData()
print(dataFrame)
# dataFrameLong = changeFormat(dataFrame)
# print(dataFrameLong)
transposedDataFrame = transposeDataFrame(dataFrame)
ctshData = getCtshData(transposedDataFrame)
bkngData = getBkngData(transposedDataFrame)
regnData = getRegnData(transposedDataFrame)
msftData = getMsftData(transposedDataFrame)
monthlyDataFrame = getMonthlyData(dataFrame)
# for stock in chosenStocks:
#     linePlot(stock, monthlyDataFrame)
# stockSeries = [ctshData, bkngData, regnData, msftData]
# for stock in stockSeries:
#     univariateAnalysis(stock)
# # swarmPlot(monthlyDataFrame)
# kernelDensityPlot(dataFrame)
# scaledData = standardizeData(transposedDataFrame)
# dataPca = reduceData(scaledData)
# applyClustering(dataPca)
# correlation = findCorrelation(dataFrame)
# rankCorrelation(correlation)
# plotHeatmap(correlation)


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