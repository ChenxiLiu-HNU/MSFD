import csv
import math
import os
import shutil
from math import asin, cos, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
from scipy.optimize import curve_fit
from sqlalchemy import create_engine

PATH_WORK = "F:/大论文/实验/Data/sz_work"
PATH_REST = "F:/大论文/实验/Data/sz_rest"
PATH_IN   = "F:/大论文/实验/Data/sz_work"
PATH_IN2  = "F:/大论文/实验/Data/sz_work_经纬度相连"
# PATH_IN1   = "F:/大论文/实验/Data/规律车辆id/sz_work_规律"
# PATH_IN2  = "F:/大论文/实验/Data/规律车辆id/sz_work_经纬度相连_规律id"
PATH_OUT  = "F:/大论文/实验/Data/sz_id~"  #剔除行程小于0.5km的数据


def distance(lat1,lon1, lat2,lon2):
    """计算两经纬度之间的距离,返回值单位：km"""
    r = 6371.0088
    def radians(d):
        return d * math.pi / 180.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    distance = c * r
    return distance

PATH = "F:/大论文/实验/Data/sz_id"
filenames = os.listdir(PATH)
v = 2
for filename in filenames:
    print("processsing:",filename)
    file = os.path.join(PATH,filename)
    temp = pd.read_csv(file,index_col=0)
    x1 = len(temp)
    for i in range(1, len(temp) + 1):

        s1 = temp.loc[i, 'Distance'] / temp.loc[i, 'TravelTime']
        if (s1 > v):
            temp = temp.drop([i], axis=0)
            print(i)
    x2 = len(temp)

    print(x1, x2)



# for i in range(2,len(temp)+1):
# #     if(temp.loc[i,'Speed']==0 and temp.loc[i-1,'Speed']==0):
# #         temp.loc[i,'Lon'] = (temp.loc[i,'Lon']+temp.loc[i-1,'Lon'])/2
# #         temp.loc[i,'Lat'] = (temp.loc[i,'Lat']+temp.loc[i-1,'Lat'])/2
# #         temp = temp.drop([i-1],axis=0)
# # x2 = len(temp)
# # print(x1,x2)

"""
temp = temp.set_index(pd.Index(np.array(range(len(temp))) + 1))
temp['GPSTime'] = pd.to_datetime(temp['GPSTime'])
temp['Day'] = temp['GPSTime'].dt.day
temp['Hour'] =temp['GPSTime'].dt.hour
temp['Min'] = temp['GPSTime'].dt.minute
temp['Second'] = temp['GPSTime'].dt.second

for i in range(1,len(temp)+1):
    temp.loc[i,'Time'] = temp.loc[i,'Day']*24+temp.loc[i,'Hour']+temp.loc[i,'Min']/60+temp.loc[i,'Second']/3600

v = 120

for i in range(2,len(temp)):

    s1 = (distance(temp.loc[i,'Lat'],temp.loc[i,'Lon'],temp.loc[i-1,'Lat'],temp.loc[i-1,'Lon']))/(temp.loc[i,'Time']-temp.loc[i-1,'Time'])
    s2 = (distance(temp.loc[i,'Lat'],temp.loc[i,'Lon'],temp.loc[i+1,'Lat'],temp.loc[i+1,'Lon']))/(temp.loc[i+1,'Time']-temp.loc[i,'Time'])
    if(s1>v and s2>v):
        print(i)
        temp = temp.drop([i],axis=0)
x3 = len(temp)

print(x1,x2,x3)
"""
"""
filenames = os.listdir(PATH)
filenames.sort(key=lambda x: int(x[:-4]))
for filename in filenames:
    #print("processing:",filename)
    file = os.path.join(PATH,filename)
    temp = pd.read_csv(file,index_col=0)
        #print(temp.columns.values.tolist())
    temp_0 = temp[temp['Speed']==0]

        #print(temp_0)
    if(len(temp_0)!=len(temp)):
        print(filename,len(temp_0),len(temp))
    #print(len(temp_0),len(temp))
"""

"""
filenames = os.listdir(PATH_IN2)
filenames.sort(key=lambda x: int(x[:-4]))
# for filename in filenames:
#         print(filename)
        # src_file = os.path.join(PATH_IN,filename)
        # des_file = os.path.join(PATH_IN1,filename)
        # shutil.copy(src_file,des_file)


for filename in filenames:
        file = os.path.join(PATH_IN,filename)
        file2 = os.path.join(PATH_IN2,filename)
        print("processing:",file)
        temp = pd.read_csv(file,index_col=0)
        temp2 = pd.read_csv(file2,index_col=0)
        x = len(temp)
        index_list = temp2.columns.values.tolist()
        print(index_list)
        if 'labels' in index_list:
                for i in range(1,x+1):
                        slabel = temp2.loc[2*i-1,'Tlabels']
                        temp.loc[i,'STlabel']  = slabel
                        elabel = temp2.loc[2*i,'Tlabels']
                        temp.loc[i,'ETlabel']  = elabel
                        temp.to_csv(file)
        else:
                des_file = os.path.join(PATH_OUT,filename)
                shutil.move(file2,des_file)


filenames = os.listdir(PATH_IN)
filenames.sort(key=lambda x: int(x[:-4]))
for filename in filenames:
    file = os.path.join(PATH_IN,filename)
    print("processing:",filename)
    temp = pd.read_csv(file,index_col=0)
    #temp.rename(columns={'DuringTime' : 'duringTime'}, inplace=True)
    #temp.rename(columns={'duringRange':'DuringRange'}, inplace=True)
    #temp['sTime'] = pd.to_datetime(temp['sTime'])
    temp['StartTime'] = pd.to_datetime(temp['StartTime'])
    temp['Month'] = temp['StartTime'].dt.month
    temp['Day'] = temp['StartTime'].dt.day
    temp['IsSunday'] = [0]*len(temp)
    temp['StartTime'] = pd.to_datetime(temp['StartTime'])

    for i in range(1,len(temp)+1):
        if temp.loc[i,'Month'] == 6:
            if(temp.loc[i,'Day'] == 5
                    or temp.loc[i,'Day'] == 12
                    or temp.loc[i, 'Day'] == 19
                    or temp.loc[i, 'Day'] == 26):
                temp.loc[i,'IsSunday'] = 1
                print(i)
        if temp.loc[i,'Month'] == 7:
            if (temp.loc[i, 'Day'] == 3
                    or temp.loc[i, 'Day'] == 10
                    or temp.loc[i, 'Day'] == 17
                    or temp.loc[i, 'Day'] == 24
                    or temp.loc[i, 'Day'] == 31):
                temp.loc[i, 'IsSunday'] = 1
                print(i)
        if temp.loc[i,'Month'] == 8:
            if (temp.loc[i, 'Day'] == 7
                    or temp.loc[i, 'Day'] == 14
                    or temp.loc[i, 'Day'] == 21
                    or temp.loc[i, 'Day'] == 28):
                temp.loc[i, 'IsSunday'] = 1
        #print(temp.columns.values.tolist())
        day = float((temp.loc[i, 'DuringTime'].split(" days ", 1))[0])
        time = (temp.loc[i, 'DuringTime'].split(" days ", 1))[1]
        hour = float((time.split(":", 2))[0])
        min = float((time.split(":", 2))[1])
        seconds = float((time.split(":", 2))[2])
        SpentTime = day * 24 * 60 + hour * 60 + min + seconds / 60
        temp.loc[i,'DuringTime'] = SpentTime
        #print("第" + str(i) + "行: ", temp.loc[i, 'DuringTime'],temp.loc[i,'IsSunday'])
    temp.rename(columns={'DuringTime': 'TravelTime'}, inplace=True)

    temp_work = temp[temp['IsSunday']==0]
    temp_work = temp_work.drop(['Month','Day'],axis=1)
    temp_work = temp_work.set_index(pd.Index(np.array(range(len(temp_work))) + 1))
    temp_rest = temp[temp['IsSunday']==1]
    temp_rest = temp_rest.drop(['Month', 'Day'], axis=1)
    temp_rest = temp_rest.set_index(pd.Index(np.array(range(len(temp_rest))) + 1))
    file_work = os.path.join(PATH_WORK,filename)
    file_rest = os.path.join(PATH_REST,filename)
    temp_work.to_csv(file_work)
    temp_rest.to_csv(file_rest)
    #temp.to_csv(file)
"""
"""
record = pd.DataFrame(index=['ObjectID','Day','Frequency'])
print(record)
i = 0
file_out = os.path.join(PATH_OUT, '深圳市车辆数据统计表_工作日.csv')
f = open(file_out, 'a', encoding='utf-8',newline='')
csv_writer = csv.writer(f)
f.close()


# file_out = os.path.join(PATH_OUT, '车辆数据统计表_工作日.csv')
# temp = pd.read_csv(file_out)
# print(len(temp[temp.Day>=30]))

PATH_IN = PATH_WORK
filenames = os.listdir(PATH_IN)
filenames.sort(key=lambda x: int(x[:-4]))
for filename in filenames:

    file = os.path.join(PATH_IN,filename)
    print("processing:",file)
    temp = pd.read_csv(file,index_col=0)
    print(temp.columns.values.tolist())
    if(len(temp)==0):
        os.remove(file)
    else:
        temp['StartTime'] = pd.to_datetime(temp['StartTime'])
        Day = len(set(temp['StartTime'].dt.day))
        Fre = len(temp)//Day
        f1 = open(file_out, 'a', encoding='utf-8', newline='')
        csv.writer(f1).writerow([filename[:-4],Day,Fre])
        f1.close()

# record = pd.DataFrame(index=['ObjectID','Day','Frequency'])
# print(record)
# i = 0
# file_out = os.path.join(PATH_OUT, '车辆数据统计表_周末.csv')
# f = open(file_out, 'a', encoding='utf-8',newline='')
# csv_writer = csv.writer(f)
# f.close()


# PATH_IN = PATH_REST
# filenames = os.listdir(PATH_IN)
# filenames.sort(key=lambda x: int(x[:-4]))
# for filename in filenames:
#     file = os.path.join(PATH_IN,filename)
#     print("processing:",file)
#     temp = pd.read_csv(file,index_col=0)
"""
"""
    if (len(temp) == 0):
        os.remove(file)
    else:
        temp['StartTime'] = pd.to_datetime(temp['StartTime'])
        Day = len(set(temp['StartTime'].dt.day))
        Fre = len(temp)//Day
        f1 = open(file_out, 'a', encoding='utf-8', newline='')
        csv.writer(f1).writerow([filename[:-4],Day,Fre])
        f1.close()
"""
# data.rename(columns={'DuringTime':'TravelTime'},inplace=True)
# print(data.columns.values.tolist())
