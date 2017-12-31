import pandas as pd
import numpy as np

ar_df = pd.read_csv('csv/air_reserve.csv')
asi_df = pd.read_csv('csv/air_store_info.csv')
avd_df = pd.read_csv('csv/air_visit_data.csv')
hr_df = pd.read_csv('csv/hpg_reserve.csv')
hsi_df = pd.read_csv('csv/hpg_store_info.csv')
sir_df = pd.read_csv('csv/store_id_relation.csv')
sample_df = pd.read_csv('csv/sample_submission.csv')
di_df = pd.read_csv('csv/date_info.csv')


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

# constrcut hpg-air genre map
genre = rrvf_df[['air_genre_name', 'hpg_genre_name']]
union_genre = genre[genre['air_genre_name'] != 0][genre['hpg_genre_name'] != 0]
group_genre = union_genre.groupby(['hpg_genre_name','air_genre_name']).size()
genre_map = []
for hpg_genre in group_genre.index.levels[0]:
    target_air_genre = group_genre[hpg_genre].argmax()
    if hpg_genre == 'Italian':
        target_air_genre = 'Italian/French'
    genre_map.append([hpg_genre, target_air_genre])
genre_map = np.array(genre_map)

# process genre
genre = []
for index, row in rrvf_df.iterrows():
    air_genre = row['air_genre_name']
    hpg_genre = row['hpg_genre_name']
    if air_genre != 0:
        genre.append(air_genre)
    elif air_genre == 0 and hpg_genre != 0:
        target_posi = np.argwhere(genre_map == hpg_genre)
        if target_posi.size != 0:
            genre_name = genre_map[target_posi[0][0], 1]
            genre.append(genre_name)
        else:
            genre.append(hpg_genre)
    else:
        pass
genre_df = pd.DataFrame(genre, columns=['genre'])
rrvf_df = rrvf_df.join(genre_df)
rrvf_df.drop(['air_genre_name', 'hpg_genre_name'], axis=1, inplace=True)

# process area_name
area_name = []
for index, row in rrvf_df.iterrows():
    air_area_name = row['air_area_name']
    hpg_area_name = row['hpg_area_name']
    if air_area_name != 0:
        area_name.append(air_area_name)
    elif air_area_name == 0 and hpg_area_name != 0:
        area_name.append(hpg_area_name)
    else:
        pass
area_name_df = pd.DataFrame(area_name, columns=['area_name'])
rrvf_df = rrvf_df.join(area_name_df)
rrvf_df.drop(['air_area_name', 'hpg_area_name'], axis=1, inplace=True)

rrvf_df.to_csv('rrvf.csv', encoding='utf-8')
