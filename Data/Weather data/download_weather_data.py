import requests
import pandas as pd
from unidecode import unidecode
import time

start_date = '2014-01-01'
end_date = '2022-12-31'
#end_date = '2014-01-03'

lat_long = pd.read_csv('lat_long_provinces.csv')

weather_data = pd.DataFrame()
for i in range(len(lat_long)):
#for i in range(3):  # para pruebas

    province = unidecode(lat_long.iloc[i,1].replace(" ", "")[:4])
    print('Starting download for',province,'-----------')
    start_time = time.time()
    
    latitude = lat_long.iloc[i,2]
    longitude = lat_long.iloc[i,3]
    uri = f'https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=direct_radiation,windspeed_10m&timezone=Europe%2FBerlin'
    
    response = requests.get(uri)
    data = response.json()
    data_dict = {
        'date': data['hourly']['time'],
        'w_' + province: data['hourly']['windspeed_10m'],
        'r_' + province: data['hourly']['direct_radiation']
    }
    
    df = pd.DataFrame(data_dict)
    if 'date' not in weather_data.columns:
        weather_data = df.copy()
    else:
        weather_data = weather_data.merge(df[['date', 'w_'+province, 'r_'+province]], on='date', how='outer')
    end_time = time.time()
    print('  time:', round(end_time - start_time),'s')
    
weather_data = weather_data[['date'] + sorted(weather_data.columns[1:])].copy()

# Separate day and hour in new cols
weather_data = weather_data.rename(columns={'date': 'date0'})
weather_data['date0'] = pd.to_datetime(weather_data['date0'])
weather_data.insert(0, 'date', weather_data['date0'].dt.strftime('%Y%m%d').astype(int))
weather_data.insert(1, 'hour', weather_data['date0'].dt.hour.astype(int)+1)
weather_data.drop(columns=['date0'], inplace=True)

weather_data.to_csv('weather_data_historical.csv', index=False)