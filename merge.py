"""
MERGE
"""
from datetime import datetime

import pandas as pd

print('MERGE: IMPORT COMPLETE! START PROCESSING...\n')


def _pre_ahdn():
    print('ahdn start!')
    ahdn_df = pd.read_csv('air+hpg_day_new.csv')

    ahdn_df.rename(columns={'time': 'visit_datetime'}, inplace=True)
    ahdn_df = pd.melt(
        ahdn_df,
        id_vars=['visit_datetime'],
        var_name='genre',
        value_name='genre_reserve_visitors')
    ahdn_df['visit_datetime'] = ahdn_df['visit_datetime'].apply(
        lambda string: datetime.strptime(string, '%Y/%m/%d').date())

    return ahdn_df


def _pre_bigAreaR():
    print('bigAreaR start!')
    bigAreaR_df = pd.read_csv('Big_area_date_rev.csv', encoding='gbk')

    bigAreaR_df = (bigAreaR_df.drop(['Unnamed: 0'], axis=1).rename(
        columns={
            'visit_date': 'visit_datetime',
            '?saka-fu_visitors': 'Ōsaka-fu_visitors'
        }))
    bigAreaR_df = pd.melt(
        bigAreaR_df,
        id_vars=['visit_datetime'],
        var_name='big_area',
        value_name='big_area_reserve_visitors')
    bigAreaR_df['big_area'] = bigAreaR_df['big_area'].apply(
        lambda name: name[:-9])
    bigAreaR_df['visit_datetime'] = bigAreaR_df['visit_datetime'].apply(
        lambda d: datetime.strptime(d, '%Y-%m-%d').date())
    bigAreaR_df = bigAreaR_df.fillna(method='ffill')
    bigarea_res = bigAreaR_df['big_area_reserve_visitors']
    bigAreaR_df['big_area_reserve_visitors'] = bigarea_res.astype('int64')

    return bigAreaR_df


def _pre_cdr():
    print('cdr start!')
    cdr_df = pd.read_csv('cluster_date_rev.csv')

    cdr_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    cdr_df = pd.melt(
        cdr_df,
        id_vars=['visit_datetime'],
        var_name='ll_cluster',
        value_name='ll_cluster_reserve_visitors')
    cdr_df['visit_datetime'] = cdr_df['visit_datetime'].apply(
        lambda string: datetime.strptime(string, '%Y-%m-%d').date())

    return cdr_df


def _pre_store():
    print('store start!')
    store_df = pd.read_csv('store_info.csv')

    cluster_code = "ABCDEFGHIJK"
    store_df['ll_cluster'] = store_df['ll_cluster'].apply(
        lambda c: cluster_code[c])

    return store_df


def _pre_apr():
    print('apr start!')
    apr_df = pd.read_csv('air_proc_res.csv')

    apr_df['visit_datetime'] = apr_df['visit_datetime'].apply(
        lambda d: datetime.strptime(d, '%Y-%m-%d').date())
    apr_df = apr_df.fillna(method='ffill')
    apr_df['reserve_visitors'] = apr_df['reserve_visitors'].astype('int64')
    apr_df = apr_df.rename(columns={
        'reserve_visitors': 'air_store_reserve_visitors'
    })

    return apr_df


def _pre_hpr():
    print('hpr start!')
    hpr_df = pd.read_csv('hpg_proc_res.csv')

    hpr_df['visit_datetime'] = hpr_df['visit_datetime'].apply(
        lambda d: datetime.strptime(d, '%Y-%m-%d').date())
    hpr_df = hpr_df.fillna(method='ffill')
    hpr_df['reserve_visitors'] = hpr_df['reserve_visitors'].astype('int64')
    hpr_df = hpr_df.rename(columns={
        'reserve_visitors': 'hpg_store_reserve_visitors'
    })

    return hpr_df


def _pre_train(store_df):
    print('train start!')
    sir_df = pd.read_csv('csv/store_id_relation.csv')

    train_df = store_df.drop(['latitude', 'longitude'], axis=1)
    train_area_df = pd.DataFrame(
        train_df['area_name'].str.split(' ', 2).tolist(),
        columns=['big_area', 'mid_area', 'sml_area'])
    # 因为中地域与小地域资讯未齐全，故舍弃
    train_df = train_df.join(train_area_df).drop(
        ['area_name', 'mid_area', 'sml_area'], axis=1)
    cover_df = train_df[train_df['hpg_store_id'].isin(sir_df['hpg_store_id'])]
    train_df.drop(cover_df.index, inplace=True)

    hpg_df = train_df[train_df['hpg_store_id'] != '0']
    air_df = train_df[train_df['air_store_id'] != '0']

    return [air_df, cover_df, hpg_df]


def _merge_foreginer(dfs, foreginers):
    # merge_list = [('hpg_store_id', hpr_df), ('air_store_id', apr_df),
    #               ('big_area', bigAreaR_df)]
    ahdn_df = foreginers['genre']
    tr_dfs = []
    # print(ahdn_df)
    for df in dfs:
        # print(df.head())
        df = pd.merge(df, ahdn_df, on=['genre'], how='inner')
        for target, target_df in foreginers.items():
            if (target == 'genre'):
                continue
            df = pd.merge(
                df, target_df, on=[target, 'visit_datetime'], how='left')
        tr_dfs.append(df)

    return tr_dfs


def _dec_air(air_df):
    air_df = (air_df.drop(
        ['hpg_store_reserve_visitors'],
        axis=1).rename(columns={
            'air_store_reserve_visitors': 'store_reserve_visitors'
        }))
    air_store_rev_mean = air_df['store_reserve_visitors'].mean()
    air_df.fillna(
        value={'store_reserve_visitors': air_store_rev_mean}, inplace=True)

    return air_df


def _dec_hpg(hpg_df):
    hpg_df = (hpg_df.drop(
        ['air_store_reserve_visitors'],
        axis=1).rename(columns={
            'hpg_store_reserve_visitors': 'store_reserve_visitors'
        }))
    hpg_store_rev_mean = hpg_df['store_reserve_visitors'].mean()
    hpg_df.fillna(
        value={'store_reserve_visitors': hpg_store_rev_mean}, inplace=True)

    return hpg_df


def _dec_cover(cover_df):
    cas_res_vis = cover_df['air_store_reserve_visitors']
    chs_res_vis = cover_df['hpg_store_reserve_visitors']
    cas_rev_mean = cas_res_vis.mean()
    chs_rev_mean = chs_res_vis.mean()
    cover_df.fillna(
        value={
            'air_store_reserve_visitors': cas_rev_mean,
            'hpg_store_reserve_visitors': chs_rev_mean
        },
        inplace=True)
    cover_store_res = chs_res_vis + cas_res_vis
    cover_df = cover_df.join(
        pd.DataFrame(cover_store_res, columns=['store_reserve_visitors']))
    cover_df.drop(
        ['hpg_store_reserve_visitors', 'air_store_reserve_visitors'],
        axis=1,
        inplace=True)

    return cover_df


def output_res_train(air_df, cover_df, hpg_df):
    res_train_df = pd.concat([air_df, cover_df, hpg_df])

    # decorate train
    res_train_df = _get_dummies(res_train_df,
                                ['genre', 'big_area', 'll_cluster'])
    # print(type(res_train_df['visit_datetime'].iloc[0]))
    res_train_df['visit_datetime'] = res_train_df['visit_datetime'].apply(
        lambda d: (d - datetime(2016, 1, 1).date()).days
    )

    res_train_df.to_csv('res_train.csv', index=False)

    return res_train_df


def _get_dummies(df, dummies_list):
    dummies_train = pd.get_dummies(df[dummies_list])
    df = pd.concat([df, dummies_train], axis=1).drop(dummies_list, axis=1)
    return df


if __name__ == '__main__':
    print('pre-process foregin dataframe')
    # pre-process foregin dataframe
    ahdn_df = _pre_ahdn()
    bigAreaR_df = _pre_bigAreaR()
    cdr_df = _pre_cdr()

    print('pre-process common dataframe')
    # pre-process common dataframe
    apr_df = _pre_apr()
    hpr_df = _pre_hpr()
    store_df = _pre_store()
    dfs = _pre_train(store_df)

    print('merge foreginer into common')
    # merge foreginer into common
    tr_dfs = _merge_foreginer(
        dfs, {
            'genre': ahdn_df,
            'hpg_store_id': hpr_df,
            'air_store_id': apr_df,
            'big_area': bigAreaR_df,
            'll_cluster': cdr_df
        })
    air_df, cover_df, hpg_df = tr_dfs

    print('decorate dfs')
    # decorate dfs
    air_df = _dec_air(air_df)
    cover_df = _dec_cover(cover_df)
    hpg_df = _dec_hpg(hpg_df)

    print('final merge')
    # final merge
    res_train_df = output_res_train(air_df, cover_df, hpg_df)
    # airres_train_df = output_airres_train(air_df, cover_df)

    print('completed')
