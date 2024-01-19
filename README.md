# Welcome to my SOLFINTECH Intelligent Stock Trader application!

SOLFINTECH is an Intelligent Stock Trader capable of performing analysis on a set of 4 chosen stocks - CTSH, BKNG, REGN and MSFT - and use Machine Learning techniques to predict the future Closing performance of these stocks.

You can analyse your chosen stock using:

- Univariate analysis of stock performance in Opening Prices, Closing Prices, Price Lows, Price Highs and Trading Volume
- Multivariate analysis of stock performance in Opening Prices, Closing Prices, Price Lows, Price Highs and Trading Volume
- Buy & Sell Signals using Simple Moving Averages (Mean of 30 and mean of 100)

Make stock value predictions using:

- ARIMA
- LSTM
- Facebook Prophet
- Linear Regression
- Polynomial Regression

Other features:

- View how the stocks have been clustered using K-Means
- View the top 10 positive and negative correlations of a chosen stock

## Setting Up the Application

My application was developed using PyCharm, but you do not need it to use the program. To run the application yourself, clone the repository from GitHub and install the required dependencies in `requirements.txt` using `pip install -r requirements.txt`. You can launch the app from either PyCharm, Windows PowerShell or your Command Prompt with `streamlit run main.py`. The program will run on `localhost:8501`. To terminate the program, press `Ctrl + C` in your chosen terminal.

Please be aware that some of the features in the application can take some time to load - especially the Machine Learning models.
 
