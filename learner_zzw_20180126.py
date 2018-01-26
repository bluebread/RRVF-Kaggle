#!/usr/bin/env python
# -*-coding:utf-8 -*-

# learner.py
"""
 Copyright© 26 Jan 2018 Zheng Zangwei
 All rights reserved.
 	@Author: Zhengzangw
	@Date: 26/01/2018, 21:01
	@Last Modified time: 26/01/2018, 21:01
	@Brief Description:
        Use date,genre,city,district to predict visit
	@Input: sample_submission.csv,air_visit_data.csv,air_store_info.csv'
	@Output: sample_submission.csv,zzw_first.pkl
"""

#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle
from sklearn import svm

#read_csv
test = pd.read_csv("sample_submission.csv")
visit = pd.read_csv("air_visit_data.csv")
air_info = pd.read_csv('air_store_info.csv')

#Initial air_info
city = [x.split(' ')[0] for x in air_info.air_area_name]
district = [x.split(' ')[1] for x in air_info.air_area_name]
air_info['city'] = city
air_info['district'] = district
air_info.drop(['air_area_name','latitude','longitude'],axis=1,inplace=True)

#Initial visit
visit["visit_date"]=pd.to_datetime(visit["visit_date"])
new_col = [1 for x in visit["visit_date"]]
visit["one"] = new_col

#Merge
visit2 = visit.merge(air_info,how='inner')
visit2["visit_date"]=pd.to_datetime(visit2["visit_date"])

# Generate the sample data set
#label
visit_target = visit2[visit2["visit_date"]>=pd.to_datetime("2017-02-01")]
visit_target["date"]=visit_target["visit_date"]-pd.to_datetime("2017-02-01")
visit_target.rename(columns={'visitors':'label'},inplace=True)
visit_target.reset_index(inplace=True)
visit_target.head()

#sample
visit_sample = visit2[(visit2.visit_date>=pd.to_datetime("2016-02-03")) & (visit2.visit_date<=pd.to_datetime("2016-04-23"))]
visit_sample["visit_date"] = visit_sample["visit_date"]-pd.to_datetime("2016-02-03")
visit_sample.reset_index(inplace=True)
visit_target.drop("index",axis=1,inplace=True)

#Group by Date
Date_Group = visit_sample.groupby('visit_date')['visitors'].mean()
#Group by Date&Genre
Genre_Group = visit_sample.groupby(['visit_date','air_genre_name'])['visitors'].mean()
#Group by city
City_Group = visit_sample.groupby(['visit_date','city'])['visitors'].mean()
#Group by district
District_Group = visit_sample.groupby(['visit_date','district'])['visitors'].mean()

#Generate sample set
newcols = [Date_Group[x] for x in visit_target.date]
visit_target["Date_Group"] = newcols
newcols = [Genre_Group[visit_target.date[x]].get(visit_target.air_genre_name[x],Date_Group[visit_target.date[x]]) for x in range(visit_target.shape[0])]
visit_target["Genre_Group"] = newcols
newcols = [City_Group[visit_target.date[x]].get(visit_target.city[x],Date_Group[visit_target.date[x]]) for x in range(visit_target.shape[0])]
visit_target["City_Group"] = newcols
newcols = [District_Group[visit_target.date[x]].get(visit_target.district[x],Date_Group[visit_target.date[x]]) for x in range(visit_target.shape[0])]
visit_target["District_Group"] = newcols

#Generate learner
sample = visit_target[['Date_Group','Genre_Group','City_Group','District_Group']].values.tolist()
label = visit_target['label'].values.tolist()
clf = svm.SVR()
clf.fit(sample,label)

#Initial visit_test
visit_test = visit2[(visit2.visit_date>=pd.to_datetime("2016-04-24")) & (visit2.visit_date<=pd.to_datetime("2016-06-01"))]
visit_test["visit_date"] = visit_test["visit_date"]-pd.to_datetime("2016-04-24")
visit_test.reset_index(inplace=True)
visit_test.drop("index",axis=1,inplace=True)

#Group by Date
Date_Group_test = visit_test.groupby('visit_date')['visitors'].mean()
#Group by Date&Genre
Genre_Group_test = visit_test.groupby(['visit_date','air_genre_name'])['visitors'].mean()
#Group by city
City_Group_test = visit_test.groupby(['visit_date','city'])['visitors'].mean()
#Group by district
District_Group_test = visit_test.groupby(['visit_date','district'])['visitors'].mean()

#Initial test
date = [pd.to_datetime(x.split('_')[2])-pd.to_datetime("2017-04-23") for x in test.id]
air_id = [x.split('_')[0]+'_'+x.split('_')[1] for x in test.id]
test["date"] = date
test["air_store_id"] = air_id
new_test = test.merge(air_info,how='inner')

newcols = [Date_Group_test[x] for x in new_test.date]
new_test["Date_Group"] = newcols
newcols = [Genre_Group_test[new_test.date[x]].get(new_test.air_genre_name[x],Date_Group_test[new_test.date[x]]) for x in range(new_test.shape[0])]
new_test["Genre_Group"] = newcols
newcols = [City_Group_test[new_test.date[x]].get(new_test.city[x],Date_Group_test[new_test.date[x]]) for x in range(new_test.shape[0])]
new_test["City_Group"] = newcols
newcols = [District_Group_test[new_test.date[x]].get(new_test.district[x],Date_Group_test[new_test.date[x]]) for x in range(new_test.shape[0])]
new_test["District_Group"] = newcols
test_set = new_test[['Date_Group','Genre_Group','City_Group','District_Group']].values.tolist()

out = clf.predict(test_set)

submit = pd.read_csv("sample_submission.csv")
submit["visitors"] = pd.Series(out)

#Save
submit.to_csv('submit.csv',index=False)
joblib.dump(clf, 'zzw_first.pkl')