"""
数据集预处理
- data_process:
    将各家商店的信息进行整合。
    输出store_df，包含 air_store_id、hpg_store_id、latitude、longitude、genre、area_name。
    store_df 将另外输出成 store_info.csv。
    若只被单边数据集登录，则该商店 id 信息将补 0 。
- clustering:
    依据类型分组统计。
    输出genre_df、city_df、lati_long_df。
"""
import pandas as pd
import numpy as np

print('IMPORT COMPLETE! START PROCESSING...\n')


def store_process():
    """各家商店的资讯进行整合。"""
    store_df = _generate()
    store_df.to_csv('store_info.csv', encoding='utf-8', index=False)
    return store_df


def _generate():
    """合并 air 与 hpg 数据集，产生初步的商店信息。"""
    asi_df = pd.read_csv('csv/air_store_info.csv')
    hsi_df = pd.read_csv('csv/hpg_store_info.csv')
    sir_df = pd.read_csv('csv/store_id_relation.csv')

    # 合并 air 与 hpg 数据集
    link_df = pd.merge(asi_df, sir_df, on='air_store_id', how='outer')
    store_df = pd.merge(link_df, hsi_df, on='hpg_store_id', how='outer')
    store_df = store_df.fillna(0)

    store_df = _lati_long_process(store_df)
    store_df = _genre_porcess(store_df)
    store_df = _area_name_process(store_df)
    return store_df


def _lati_long_process(store_df):
    """处理经纬度 air 与 hpg 重叠的部份"""
    count_latitude_x = (store_df['latitude_x'] > 0) + 0
    count_latitude_y = (store_df['latitude_y'] > 0) + 0
    count_latitude = count_latitude_x + count_latitude_y
    total_latitude = store_df['latitude_x'] + store_df['latitude_y']

    count_longitude_x = (store_df['longitude_x'] > 0) + 0
    count_longitude_y = (store_df['longitude_y'] > 0) + 0
    count_longitude = count_longitude_x + count_longitude_y
    total_longitude = store_df['longitude_x'] + store_df['longitude_y']

    store_df.drop(
        ['latitude_x', 'latitude_y', 'longitude_x', 'longitude_y'],
        axis=1,
        inplace=True
    )

    ll_df = pd.DataFrame({
        'latitude': total_latitude / count_latitude,
        'longitude': total_longitude / count_longitude
    })
    store_df = pd.concat([store_df, ll_df], axis=1)

    return store_df


def _genre_porcess(store_df):
    """处理商店种类信息。因为 hpg 商店种类杂乱，主要以 air 为准。"""

    # 依照两边数据集都有登录的商店为准，归纳 hpg 能转换成为的 air 商店种类。
    genre = store_df[['air_genre_name', 'hpg_genre_name']]
    union_genre = genre[genre['air_genre_name'] != 0][genre['hpg_genre_name'] != 0]
    group_genre = union_genre.groupby([
        'hpg_genre_name',
        'air_genre_name'
        ]).size()
    genre_map = []
    for hpg_genre in group_genre.index.levels[0]:
        target_air_genre = group_genre[hpg_genre].argmax()
        # 'Italian' 明显对应 air 中 'Italian/French'，所以个别处理。
        if hpg_genre == 'Italian':
            target_air_genre = 'Italian/French'
        genre_map.append([hpg_genre, target_air_genre])
    genre_map = np.array(genre_map)

    # 进行商店种类转换
    genre = []
    for index, row in store_df.iterrows():
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
    store_df = store_df.join(genre_df)
    store_df.drop(['air_genre_name', 'hpg_genre_name'], axis=1, inplace=True)

    return store_df


def _area_name_process(store_df):
    """处理商店地域信息。以 air 登录的区域优先。"""
    area_name = []
    for index, row in store_df.iterrows():
        air_area_name = row['air_area_name']
        hpg_area_name = row['hpg_area_name']
        if air_area_name != 0:
            area_name.append(air_area_name)
        elif air_area_name == 0 and hpg_area_name != 0:
            area_name.append(hpg_area_name)
        else:
            pass
    area_name_df = pd.DataFrame(area_name, columns=['area_name'])
    store_df = store_df.join(area_name_df)
    store_df.drop(['air_area_name', 'hpg_area_name'], axis=1, inplace=True)

    return store_df


def clustering():
    """
    """
    pass
