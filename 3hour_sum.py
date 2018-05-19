# -*- coding: utf-8 -*-
"""
Created on Thu May 17 22:59:47 2018

@author: hotbr
"""
from pandas import Timestamp, DataFrame, read_csv, concat

from datetime import timedelta

DAY_DELTA = timedelta(1)
THR_HOURS_DELTA = timedelta(hours=3)

FINAL_DATE = Timestamp(2017,6,1)
START_DATE = Timestamp(2016,1,1)

AIR_ID_COL = "air_store_id"
HPG_ID_COL = "hpg_store_id"
RES_DATE_COL = "reserve_datetime"
VISIT_DATE_COL = "visit_datetime"
RES_VISITORS_COL = "reserve_visitors"

# return interval [start, start + n)
def col_interval(df, col_name, start, n):
    end = start + n
    return ((df[col_name] >= start) & (df[col_name] < end))

def thr_hour_interval_sum(res_df):
    # loop all ids
    sum_table = []
    count = 0
    for id_, id_df in res_df.groupby(AIR_ID_COL):
        count += 1
        print("{1}: {0} processing...".format(id_, count))
        visit_dates = (
                id_df[VISIT_DATE_COL]
                .apply(lambda t: t.date())
                .unique())
        # loop all dates in records
        for time_head in visit_dates:
            day_interval = col_interval(
                    id_df, VISIT_DATE_COL, time_head, DAY_DELTA)
            day_res = id_df[day_interval]
            sum_row = [id_, time_head]
            
            # count the visitors between [0,3), [3,6), ... , [21,24)
            hour_head = Timestamp(time_head)
            next_day = Timestamp(time_head + DAY_DELTA)
            while hour_head < next_day:
                hour_interval = col_interval(
                        day_res, VISIT_DATE_COL, hour_head, THR_HOURS_DELTA)
                interval_sum = day_res[hour_interval][RES_VISITORS_COL].sum()
                sum_row.append(interval_sum) # add 3-hours data
                hour_head += THR_HOURS_DELTA
                
            # add day data
            sum_table.append(sum_row)
    # output
    cols = ["id", "visit_date"]
    for i in range(8):
        col = "{0}_sum".format(3 + 3 * i)
        cols.append(col)
    DataFrame(sum_table, columns=cols).to_csv("res_sum.csv", index=False)
    print("Complete!")
    
def other_sum(df):
    qcol_names = {
            0: "nightsnack_sum", 
            1: "breakfirst_sum", 
            2: "lunch_sum", 
            3: "dinner_sum"}
    hcol_names = {0: "sun_sum", 1: "moon_sum"}
    wcol_name = "day_sum"
    # quarter sum
    qsums = []
    for i in range(4):
        col1 = "{0}_sum".format(3 + 6 * i)
        col2 = "{0}_sum".format(6 + 6 * i)
        q_sum = df[col1] + df[col2]
        qsums.append(q_sum)
    qdf = concat(qsums, axis=1).rename(columns=qcol_names)
    # half sum
    sun_col = qdf["breakfirst_sum"] + qdf["lunch_sum"]
    moon_col = qdf["dinner_sum"] + qdf["nightsnack_sum"]
    hdf = concat([sun_col, moon_col], axis=1).rename(columns=hcol_names)
    # whole sum
    wdf = qdf.sum(axis=1).to_frame(wcol_name)
    # output
    df = concat([df, qdf, hdf, wdf], axis=1)
    df.to_csv("air_res_sum.csv", index=False)
    print("Complete!")
    
 
if __name__ == '__main__':
#    ares_df = read_csv(
#            "csv/air_reserve.csv", 
#            parse_dates=["reserve_datetime", "visit_datetime"])
#    hres_df = read_csv(
#            "csv/hpg_reserve.csv", 
#            parse_dates=["reserve_datetime", "visit_datetime"])
#    thr_hour_interval_sum(ares_df)
    aressum_df = read_csv("air_res_sum.csv", parse_dates=["date"])
#    hressum_df = read_csv("hpg_res_sum.csv", parse_dates=["date"])
    other_sum(aressum_df)