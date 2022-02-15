# This project was created for analysis of Microsoft stock returns distribution
# on the base of timeseries data MSFTPrices.csv

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import shapiro

# Read in the csv file and parse dates
StockPrices = pd.read_csv('MSFTPrices.csv', parse_dates=['Date'])

# Ensure the prices are sorted by Date
StockPrices = StockPrices.sort_values(by='Date')

# Print only the first five rows of StockPrices
# print(StockPrices.head())

# Calculate the daily returns of the adjusted close price
StockPrices['Returns'] = StockPrices['Adjusted'].pct_change()

# Check the first five rows of StockPrices
print(StockPrices.head())

# Plot the returns column over time
StockPrices['Returns'].plot()
plt.show()

# Convert the decimal returns into percentage returns
percent_return = StockPrices['Returns']*100

# Drop the missing values
returns_plot = percent_return.dropna()

# Plot the returns histogram
plt.hist(returns_plot, bins=75, density=False)
plt.show()
print()
# Calculate the average daily return of the stock
mean_return_daily = np.mean(StockPrices['Returns'])
print('STOCK RETURN characteristics: mean, volatility, varience ')
print('Average daily return of the stock = ', mean_return_daily)

# Calculate the implied annualized average return
mean_return_annualized = ((1+mean_return_daily)**252)-1
print('Annualized average return = ', mean_return_annualized)

# Calculate the standard deviation of daily return of the stock
sigma_daily = np.std(StockPrices['Returns'])
print('Standard deviation (volatility) of daily stock return = ', sigma_daily)

# Calculate the daily variance
variance_daily = np.var(StockPrices['Returns'])
print('Daily stock variance = ', variance_daily)

# Annualize the standard deviation
sigma_annualized = sigma_daily*np.sqrt(252)
print('Annualized standard deviation = ', sigma_annualized)

# Calculate the annualized variance
variance_annualized = sigma_annualized**2
print('Annualized varience = ', variance_annualized)

# Drop the missing values
clean_returns = StockPrices['Returns'].dropna()
print()
# Calculate the third moment (skewness) of the returns distribution
returns_skewness = skew(clean_returns)
print('SKEWNESS and KURTOSIS of stock return distribution')
print('Skewness = ', returns_skewness)
if returns_skewness > 0.0:
    print('Sign of non-normality of distribution.')
else:
    print('Distribution is normal.')


# Calculate the excess kurtosis of the returns distribution
excess_kurtosis = kurtosis(StockPrices['Returns'].dropna())
print('Excess kurtosis = ', excess_kurtosis)

# Derive the true fourth moment of the returns distribution
fourth_moment = excess_kurtosis + 3
print('Kurtosis of the returns distribution = ', fourth_moment)
if fourth_moment > 3.0:
    print('Sign of non-normality of distribution.')
else:
    print('Distribution is normal.')
print()

# Run the Shapiro-Wilk test on the stock returns
shapiro_results = shapiro(clean_returns)
print("Shapiro-Wilk test results:", shapiro_results)

# Extract the p-value from the shapiro_results
p_value = shapiro_results[1]
print("P-value: ", p_value)
if p_value <=0.05:
    print('Null hypothesis of normality is rejected.')
else:
    print('Null hypothesis of normality is accepted.')