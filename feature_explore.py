import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ar_df = pd.read_csv('csv/air_reserve.csv')
asi_df = pd.read_csv('csv/air_store_info.csv')
avd_df = pd.read_csv('csv/air_visit_data.csv')
hr_df = pd.read_csv('csv/hpg_reserve.csv')
hsi_df = pd.read_csv('csv/hpg_store_info.csv')
sir_df = pd.read_csv('csv/store_id_relation.csv')
sample_df = pd.read_csv('csv/sample_submission.csv')
di_df = pd.read_csv('csv/date_info.csv')

# longitude-latitude cluster
xh = hsi_df['longitude']
yh = hsi_df['latitude']
plt.scatter(xh,yh)
plt.show()

xa = asi_df['longitude']
ya = asi_df['latitude']
plt.scatter(xa,ya)
plt.show()

# location counting
asi_location_df = pd.DataFrame(asi_df.air_area_name.str.split(' ', 2).tolist(), columns = ['province','city', 'street'])
asi_df = asi_df.join(asi_location_df)
asi_df.drop(['air_area_name'], axis=1, inplace=True)

hpg_location_df = pd.DataFrame(hsi_df.hpg_area_name.str.split(' ', 2).tolist(), columns = ['province','city', 'street'])
hsi_df = hsi_df.join(hpg_location_df)
hsi_df.drop(['hpg_area_name'], axis=1, inplace=True)

province_count = pd.concat([asi_df['province'], hsi_df['province']]).value_counts().size # 13
city_count = pd.concat([asi_df['city'], hsi_df['city']]).value_counts().size # 85
street_count = pd.concat([asi_df['street'], hsi_df['street']]).value_counts().size # 191
location_total = province_count + city_count + street_count # 289
print('location_total:', location_total)

# merge air_store_info & hpg_store_info dataset
link_df = pd.merge(asi_df, sir_df, on='air_store_id', how='outer')
rrvf_df = pd.merge(link_df, hsi_df, on='hpg_store_id', how='outer')
rrvf_df = rrvf_df.fillna(0)
rrvf_df.head()

# process latitude & longitude overlapping
count_latitude_x = (rrvf_df['latitude_x'] > 0) + 0
count_latitude_y = (rrvf_df['latitude_y'] > 0) + 0
count_latitude = count_latitude_x + count_latitude_y
mean_latitude = (rrvf_df['latitude_x'] + rrvf_df['latitude_y']) / count_latitude

count_longitude_x = (rrvf_df['longitude_x'] > 0) + 0
count_longitude_y = (rrvf_df['longitude_y'] > 0) + 0
count_longitude = count_longitude_x + count_longitude_y
mean_longitude = (rrvf_df['longitude_x'] + rrvf_df['longitude_y']) / count_longitude

rrvf_df.drop(['latitude_x', 'latitude_y', 'longitude_x', 'longitude_y'], axis=1, inplace=True)

ll_df = pd.DataFrame({'latitude': mean_latitude, 'longitude': mean_longitude})
rrvf_df.join(ll_df)

# counting continuous holiday duration
record_arr = [['2016-01-01', 0]]
record_posi = 0
recording = True
for index, row in di_df.iterrows():
    if recording and row['holiday_flg'] is 1:
        record_arr[record_posi][1] += 1
    elif not recording and row['holiday_flg'] is 1:
        record_posi += 1
        record_arr.append([row['calendar_date'], 1])
        recording = True
    elif recording and row['holiday_flg'] is not 1:
        recording = False
    else:
        pass