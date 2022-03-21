# This project was created for analysis of the 9 biggest companies portfolio
# on the base of timeseries data Big9Returns2017.csv

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the csv file and parse dates
StockReturns = pd.read_csv('Big9Returns2017.csv')

# Inspect data
# print(StockReturns.info())

# Convert the date column to datetime64
StockReturns.Date=pd.to_datetime(StockReturns.Date)

# Set date column as index
StockReturns.set_index('Date', inplace=True)

# Inspect data
# print(StockReturns.info())

# Plot StockReturns
# StockReturns.plot(subplots=True)
# plt.show()

# Ensure the prices are sorted by Date
StockReturns = StockReturns.sort_values(by='Date')

# Print only the first five rows of StockReturns
# print(StockReturns.head())

# Finish defining the portfolio weights as a numpy array
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# Calculate the weighted stock returns
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)

# Calculate the portfolio returns
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)
# Calculate cumulative portfolio returns over time
StockReturns["Portfolio"] = ((1+WeightedReturns.sum(axis=1)).cumprod()-1)

# How many stocks are in your portfolio?
numstocks = 9
# Create an array of equal weights across all assets
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)
# Calculate the equally-weighted portfolio returns
EqualyWeightedReturns = StockReturns.iloc[:, 0 : 9].mul(portfolio_weights_ew, axis=1).sum(axis=1)
# Calculate cumulative equally-weighted portfolio returns over time
StockReturns["Portfolio_EW"] = ((1+EqualyWeightedReturns).cumprod()-1)

# Market-capitalizations weighted portfolio
# Create an array of market capitalizations (in billions)
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])
# Calculate the market cap weights
mcap_weights = market_capitalizations/sum(market_capitalizations)

# Calculate the market cap weighted portfolio returns
StockReturns['Portfolio_MCap'] = StockReturns.iloc[:, 0:9].mul(mcap_weights, axis=1).sum(axis=1)
# Calculate cumulative market cap weighted portfolio returns over time
StockReturns["Portfolio_MCap"] = ((1+StockReturns["Portfolio_MCap"]).cumprod()-1)


# Plot cumulative portfolio returns StockReturns['Portfolio']
# and cumulative equally-weighted portfolio returns StockReturns['Portfolio_EW'] over time
StockReturns['Portfolio'].plot()
StockReturns["Portfolio_EW"].plot(title='Cumulative Returns over time')
StockReturns['Portfolio_MCap'].plot()
plt.legend()
plt.show()

# CORRELATION and CO-VARIENCE

# Calculate the correlation matrix
df = pd.DataFrame(StockReturns.iloc[:,0:9])
correlation_matrix = df.corr()
# Print the correlation matrix
print(correlation_matrix)
# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu",
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Calculate the covariance matrix
cov_mat = df.cov()

# Annualize the co-variance matrix
cov_mat_annual = cov_mat*252

# Print the annualized co-variance matrix
print(cov_mat_annual)

# Calculate the portfolio standard deviation
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
print('portfolio_volatility=', portfolio_volatility)