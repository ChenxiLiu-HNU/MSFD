import os
import pandas as pd
import numpy as np
import math
import csv
import time
from math import asin, cos, sin, sqrt
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

PATH_IN       = "F:/大论文/实验/Data/sz_work_经纬度相连"
PATH_OUT      = "F:/大论文/实验"
SAME_DIST_LEN = 1 #km
r             = 6371.0088

def haversine(lon1, lat1, lon2, lat2):
    """
    将十进制转化为弧度
    """
    def rad(d):
        return d * math.pi / 180.0

    lon1, lat1, lon2, lat2 = map(rad, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
#    print("distance=", (c * r * 1000.0) / 1000.0)
    return (c * r)  # 返回距离为km

def distance(x, y):
    return haversine(x[0],x[1],y[0],y[1])


def time(x,y):
    t = x - y
    return abs(t)

"""
函数说明：聚类分析,空间
"""
def get_cluster(data_frame,ESP,MIN_SAMPLES):

    # coords = [df['sLat'],df['sLon']]
    # data = (np.array(coords))
    # x = np.transpose(data)
    """
    参数说明：
    eps:距离阈值，0.01km - 1km step=0.005km
    min_samples:样本点要成为核心对象需要的e-领域的样本数阈值
    metric:最近邻距离度量参数
    fit(x)是训练函数
    """
    #df = data_frame.drop(data_frame.columns[0], axis=1)
    #print(df)
    df = data_frame
    locs = [df['Lon'], df['Lat']]
    #print(locs)
    data = (np.array(locs))
    x = np.transpose(data)
    db = DBSCAN(eps=ESP, min_samples=MIN_SAMPLES, metric=lambda a, b: distance(a, b)).fit(x)
    labels = db.labels_
    cluster_num = len(set(labels)-set([-1]))
    #print("The number of clusters:",cluster_num)
    #统计每个簇的样本个数
    stats = str(pd.Series([i for i in labels if i != -1]).value_counts().values)
    #print(stats)
    return cluster_num,stats,labels

"""
函数说明：聚类分析,时间
"""
def get_Tcluster(data_frame,ESP,MIN_SAMPLES):

    # coords = [df['sLat'],df['sLon']]
    # data = (np.array(coords))
    # x = np.transpose(data)
    """
    参数说明：
    eps:距离阈值，0.01km - 1km step=0.005km
    min_samples:样本点要成为核心对象需要的e-领域的样本数阈值
    metric:最近邻距离度量参数
    fit(x)是训练函数
    """
    #df = data_frame.drop(data_frame.columns[0], axis=1)
    #print(df)
    df = data_frame
    locs = [df['time']]
    #print(locs)
    data = (np.array(locs))
    x = np.transpose(data)
    db = DBSCAN(eps=ESP, min_samples=MIN_SAMPLES, metric=lambda a, b: time(a, b)).fit(x)
    labels = db.labels_
    cluster_num = len(set(labels)-set([-1]))
    #print("The number of clusters:",cluster_num)
    #统计每个簇的样本个数
    stats = str(pd.Series([i for i in labels if i != -1]).value_counts().values)
    #print(stats)
    return cluster_num,stats,labels

"""
函数功能：将Stop拼接在Start后面
"""
def proOD(filename):

    file = os.path.join(PATH_IN,filename)
    print("processing:",file)
    temp = pd.read_csv(file,index_col=0)  #读取文件

    df_slon = list(temp['StartLon'])
    df_slat = list(temp['StartLat'])
    df_elon = list(temp['StopLon'])
    df_elat = list(temp['StopLat'])

    slat = df_slat+df_elat
    slon = df_slon+df_elon
    dict = {'sLon':slon,'sLat':slat}
    df = pd.DataFrame(dict)
    df = df.set_index(pd.Index(np.array(range(len(df))) + 1))
    file_out = os.path.join(PATH_OUT,filename)
    df.to_csv(file_out)

"""对时间列做处理"""
def proTime(filename):
    file = os.path.join(PATH_IN,filename)
    temp = pd.read_csv(file,index_col=0)
    temp['Time'] = pd.to_datetime(temp['Time'])
    temp['Hour'] = temp['Time'].dt.hour
    temp['Min']  = temp['Time'].dt.minute
    temp['Second'] = temp['Time'].dt.second
    temp['time'] = temp['Hour']*60+temp['Min']+temp['Second']/60
    temp = temp.drop(['Hour','Min','Second'],axis=1)
    temp.to_csv(file)

def getEps(filename):
    file = os.path.join(PATH_IN,filename)
    #print("processing:",file)
    temp = pd.read_csv(file,index_col=0)  #读取文件
    # 迭代不同的eps值
    cnum0  = len(temp)
    stats0 = []
    #for eps in np.arange(0.01, 1, 0.005):
        # 迭代不同的min_samples值
    eps=0.4
    for min_samples in range(2,30):
        cnum,stats,labels = get_cluster(temp,eps,min_samples)
        print(eps,min_samples)
        if(cnum == cnum0 and stats==stats0):  #若簇数不变，每个簇的数量不变
            temp['labels'] = labels
            temp.to_csv(file)
            return eps,min_samples,cnum
        else:
            cnum0 =cnum
            stats0 = stats
    return 0


def getTEps(filename):
    file = os.path.join(PATH_IN,filename)
    #print("processing:",file)
    temp = pd.read_csv(file,index_col=0)  #读取文件
    # 迭代不同的eps值
    cnum0  = len(temp)
    stats0 = []
    #for eps in np.arange(0.01, 1, 0.005):
        # 迭代不同的min_samples值
    eps=30
    for min_samples in range(2,30):
        cnum,stats,labels = get_Tcluster(temp,eps,min_samples)
        print(eps,min_samples)
        if(cnum == cnum0 and stats==stats0):  #若簇数不变，每个簇的数量不变
            temp['Tlabels'] = labels
            temp.to_csv(file)
            return eps,min_samples,cnum
        else:
            cnum0 =cnum
            stats0 = stats
    return 0

if __name__ == "__main__":

    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)
    filenames = os.listdir(PATH_IN)
    filenames.sort(key=lambda x: int(x[:-4]))
    #"""
    record = pd.DataFrame(index=['ObjectID','Eps','MIN_SAMPLE','Num_Cluster'])
    file_out = os.path.join(PATH_OUT, 'para_Trecord_30_sz.csv')
    f = open(file_out, 'a', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    # 3. 构建列表头
    csv_writer.writerow(["ObjectID", "EPS", "MIN_SAMPLE",'Num_Cluster'])
    f.close()
    #"""
    for filename in filenames:
        print("processing:",filename)
        #"""
        f1 = open(file_out, 'a', encoding='utf-8',newline='')
        if(getTEps(filename) != 0):
            e,m,c = getTEps(filename)
            csv.writer(f1).writerow([filename[:-4],e,m,c])
        f1.close()
