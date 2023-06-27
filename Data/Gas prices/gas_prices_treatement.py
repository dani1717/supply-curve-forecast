import pandas as pd
import datetime

# Monthly prices from 2009
month_prices = pd.read_excel('month_prices.xlsx')
month_prices.dtypes

month_prices['date'] = month_prices['MONTH'].apply(lambda x: x.strftime('%Y%m%d'))
month_prices = month_prices.drop('MONTH', axis=1)

# Daily prices from oct 2017
# https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data
daily_prices = pd.read_csv('daily_prices.csv')
daily_prices = daily_prices.iloc[:,:2]

dates = [datetime.datetime.strptime(date_str, '%m/%d/%Y') for date_str in daily_prices['Date']]
date_nums = [int(date.strftime('%Y%m%d')) for date in dates]
daily_prices['date'] = date_nums
daily_prices = daily_prices.drop('Date', axis=1)
daily_prices = daily_prices.sort_values('date', ascending=True)


date_range = pd.date_range('2014-01-01', '2022-12-31', freq='D')
dates = pd.DataFrame({'date': date_range})
dates['date'] = dates['date'].apply(lambda x: x.strftime('%Y%m%d'))
dates['date'] = dates['date'].astype(int)

gas_prices = pd.merge(dates, daily_prices, on='date', how='left')

month_prices['date_str'] = month_prices['date'].astype(str).str[:6]
month_prices = month_prices.drop('date', axis=1)
gas_prices['date_str'] = gas_prices['date'].astype(str).str[:6]
gas_prices = gas_prices.merge(month_prices, on='date_str', how='left')
gas_prices.loc[gas_prices['date'] < 20171023, 'Price'] = gas_prices.loc[gas_prices['date'] < 20171023, 'TTFprice']

gas_prices = gas_prices.iloc[:,:2]
gas_prices['Price'] = gas_prices['Price'].fillna(method='ffill')

gas_prices = gas_prices.rename(columns={'Price': 'gas_price'})

gas_prices.to_csv('gas_prices.csv', index=False)

