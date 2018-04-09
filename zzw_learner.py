import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle
from sklearn import svm

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