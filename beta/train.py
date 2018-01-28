"""
训练模型
- res_train:
    由 air 的 Visitor 与 Reserve 信息进行训练，预测
    hpg 的 Visitor 信息。
"""
from datetime import datetime

import pandas as pd
from sklearn.svm import SVR

print('TRAIN: IMPORT COMPLETE! START TRAINING...\n')


def res_train(avd_df, airrt_df, hpgrt_df):
    X, y, test_X = _pro_res(airrt_df, avd_df, hpgrt_df)

    clf = SVR()
    print('\nRES_TRAIN: START FITTING!')
    clf.fit(X, y)

    print('\nRES_TRAIN: START PREDICTING!')
    y_pred = clf.predict(test_X)

    print('\ncompleted!')
    return y_pred


def _pro_res_y():
    avd_df = pd.read_csv('csv/air_visit_data.csv')

    avd_df['visit_date'] = avd_df['visit_date'].apply(
        lambda d: datetime.strptime(d, '%Y-%m-%d').date())
    avd_df['visit_date'] = avd_df['visit_date'].apply(
        lambda d: (d - datetime(2016, 1, 1).date()).days)
    avd_df = avd_df.rename(columns={'visit_date': 'visit_datetime'})

    return avd_df


def _pro_rtrain():
    rtrain_df = pd.read_csv('res_train.csv')

    rtrain_df['air_store_id'] = rtrain_df['air_store_id'].apply(
        lambda i: '0' if i == 0 else i)
    rtrain_df['hpg_store_id'] = rtrain_df['hpg_store_id'].apply(
        lambda i: '0' if i == 0 else i)

    # airrt_df = rtrain_df[rtrain_df['air_store_id'] != '0'].drop(
    #     'hpg_store_id', axis=1)
    # hpgrt_df = rtrain_df[rtrain_df['air_store_id'] == '0'].drop(
    #     'air_store_id', axis=1)

    airrt_df = rtrain_df[rtrain_df['air_store_id'] != '0'].drop(
        ['hpg_store_id', 'big_area', 'genre', 'll_cluster'], axis=1)
    hpgrt_df = rtrain_df[rtrain_df['air_store_id'] == '0'].drop(
        ['air_store_id', 'big_area', 'genre', 'll_cluster'], axis=1)


    return airrt_df, hpgrt_df


def _pro_res(avd_df, airrt_df, hpgrt_df):
    restrain_df = pd.merge(
        airrt_df, avd_df, on=['air_store_id', 'visit_datetime'], how='inner')
    resvis_df = restrain_df['visitors']
    restrain_df.drop(
        ['air_store_id', 'big_area', 'genre', 'll_cluster', 'visit_datetime'],
        axis=1,
        inplace=True)

    X = restrain_df.values
    y = resvis_df.values
    # test_X = hpgrt_df.drop('hpg_store_id', axis=1).values

    return X, y, test_X


if __name__ == '__main__':
    # 训练前，读入数据并格式化
    avd_df = _pro_res_y()
    airrt_df, hpgrt_df = _pro_rtrain()

    # 进行训练，将最终返回结果输出成文件
    hpg_vis_pred = res_train(avd_df, airrt_df, hpgrt_df)
    hpg_vis_pred_df = pd.DataFrame(hpg_vis_pred, columns=['hpg_vis_pred'])
    hpg_vis_pred_df.to_csv('hpg_vis_pred.csv', index=True)
