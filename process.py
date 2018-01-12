"""
数据集预处理
- store_process:
    将各家商店的信息进行整合。
    返回store_df，包含 air_store_id, hpg_store_id,
    latitude, longitude, genre, area_name。
    store_df 将另外输出成 store_info.csv。
    若只被单边数据集登录，则该商店 id 信息将补 0 。
- reserve_process:
    预定信息处理。
    返回reserve_df，包含 id, visit_datetime,
    visitors，并输出 csv 檔。
- clustering:
    依据类型分组统计。
    返回genre_df、city_df、lati_long_df。
"""
from datetime import datetime
import json

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

    # 进行商店种类转换
    genre = []
    genre_map = json.load(open('genre_dict.json'))
    for index, row in store_df.iterrows():
        air_genre = row['air_genre_name']
        hpg_genre = row['hpg_genre_name']
        if air_genre != 0:
            new_genre = genre_map['air'][air_genre]
            genre.append(new_genre)
        elif air_genre == 0 and hpg_genre != 0:
            new_genre = genre_map['hpg'][hpg_genre]
            genre.append(new_genre)
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


def reserve_process():
    """
    预订信息处理，删除 reserve_datetime 并整合单日预定人数。返回具
    有 air 与 hpg 预订信息的 DataFrame，最后输出 csv 檔。
    """
    ar_df = pd.read_csv(
        'csv/air_reserve.csv',
        parse_dates=['visit_datetime', 'reserve_datetime'])
    hr_df = pd.read_csv(
        'csv/hpg_reserve.csv',
        parse_dates=['visit_datetime', 'reserve_datetime'])
    r_pro_dfs = []  # 缓存 reserve_df, 作为最后输出
    # 分别对 air 与 hpg 数据集进行处理
    for name, df in [('air', ar_df), ('hpg', hr_df)]:
        reserve_df = _r_df_process(name, df)
        csv_name = '{0}_proc_res.csv'.format(name)
        # 输出成 csv 檔
        reserve_df.to_csv(csv_name, encoding='utf-8', index=False)
        r_pro_dfs.append(reserve_df)

    return r_pro_dfs


def _r_df_process(name, df):
    """对具有预订信息的 DataFrame 进行处理。"""
    store_id_name = '{0}_store_id'.format(name)

    reserve_df = df.drop(['reserve_datetime'], axis=1)
    visit_date_convert = reserve_df['visit_datetime'].apply(
        _npdatetime64_convert_to_datetime_date)
    reserve_df['visit_datetime'] = visit_date_convert
    reserve_df = reserve_df.groupby(
        [store_id_name, 'visit_datetime'], as_index=False).sum()

    return reserve_df


def _npdatetime64_convert_to_datetime_date(npdatetime64):
    """numpy.datetime64 转换成 datetime，只传回 date 部份"""
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (npdatetime64 - unix_epoch) / one_second

    return datetime.utcfromtimestamp(seconds_since_epoch).date()


if __name__ == '__main__':
    ar_pro_df, hr_pro_df = reserve_process()
    print(ar_pro_df.head())
    print(hr_pro_df.head())
