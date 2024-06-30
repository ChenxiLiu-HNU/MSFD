import csv
import math
import os
import shutil
from math import asin, cos, sin, sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
from scipy.optimize import curve_fit
from sqlalchemy import create_engine
import PrefixSpan
import copy
import sys

PATH_MOVE = r"C:\Users\58393\Desktop\H-paper\exp\Data\sz_work"
PATH_STOP = r"C:\Users\58393\Desktop\H-paper\exp\Data\sz_work_stop"
PATH_IN = r"C:\Users\58393\Desktop\H-paper\exp\Data\sz_work_经纬度相连"
PATH_OUT  = r"C:\Users\58393\Desktop\H-paper\exp

IR_objectid = [404829,179863,380227,555761,402511,382789,123212,382754,578670,555840,
               182040,384090,118229,479839,556867,562712,574678,400072,555832,561725,
               462199,462257,467488,181610,462235,480606,578248,462178,555864,578688,
               121434,479787,181640,565764,581532,539084,539480,578240,556915,556933,
               181614,127337,382984,181665,380257,462163,460850,179904,547628,402591,
               578465,380229,444895,552592,480243,533792,181646,181669,565875,179673,
               565751,573510,121436,181628,468094,174901,403115,179666,555846,400749,
               480164,578686,382729,466488,181623,181707,181660,118258,462105,480131,
               181634,533755,543563,118184,403126,402529,382724,181699,179698,121268,
               181617,578239,31326 ,400744,462225,181621,565759,453695]

Re_ObjectId = [121266,184799,533760,551239,551241,569697,400762,402500,123474,379479,
               563053,479871,541222,533791,402526,125726,100605335,480579,543887,184768,
               547804,468379,382935,109409,382927,593187,382985,581631,384504,462923,
               126075,480012,110671,479937,474196,474802,115500,382828,117960,578473,
               537726,564055,555923,593311,462106,462152,179691,548255,179749,570855,
               468106,585194,123376,123378,462215,123466,127352,534199,477986,478064,
               556936,479987,179677,480105,379335,382909,117184,553669,126049,472527,
               563900,108831,31337,548061,116863,555930,379315,382991,461805,478082,
               478122,121394,480331,179656,136225,381046,461800,180024,555698,466774,
               548467,578258, 542936]

def get_day(temp):
    """获取行程的天数"""
    temp['StartTime'] = pd.to_datetime(temp['StartTime'])
    temp_six = temp[temp['StartTime'].dt.month == 6]
    temp_seven = temp[temp['StartTime'].dt.month == 7]
    temp_eight = temp[temp['StartTime'].dt.month == 8]
    temp_six['StartTime'] = pd.to_datetime(temp_six['StartTime'])
    temp_seven['StartTime'] = pd.to_datetime(temp_seven['StartTime'])
    temp_eight['StartTime'] = pd.to_datetime(temp_eight['StartTime'])
    # 天数
    day_len = len(set(temp_six['StartTime'].dt.day)) + \
              len(set(temp_seven['StartTime'].dt.day)) + len(set(temp_eight['StartTime'].dt.day))
    return day_len

def get_entropy(temp):
    """计算时间序列的熵率"""
    hour_list = np.arange(0,24)
    p_list = [0]*24
    temp['StartTime'] = pd.to_datetime(temp['StartTime'])
    temp['Hour'] = temp['StartTime'].dt.hour
    for i in range(1,len(temp)+1):
        for j in range(0,24):
            if(temp.loc[i,'Hour']==j):
                p_list[j] = p_list[j]+1
    for i in range(len(p_list)):
        p_list[i] = p_list[i]/len(temp)
    entropy = 0
    for i in range(len(p_list)):

        p = float(p_list[i])
        if (p > 0):
            logP = float(math.log(p, 2))
        else:
            logP = 0
        entropy = entropy + float(p * logP)

    return -entropy

def distance(lat1, lon1, lat2,lon2):

    def radians(d):
        return d * math.pi / 180.0
    r = 6371.0088
    """
    计算两经纬度之间的距离,返回值单位：km
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    distance = c * r
    return distance

def get_rg(temp):

    def getMidPoint(temp):
        sumLon = sum(temp['sLon'])
        sumLat = sum(temp['sLat'])
        # sumTime = sum(temp['SpentTime'])
        midLon = sumLon / (len(temp))
        midLat = sumLat / (len(temp))
        return midLon, midLat

    midLon,midLat = getMidPoint(temp)
    sumdist = 0
    for i in range(1, len(temp) + 1):
        sumdist = sumdist + distance(temp.loc[i, 'sLat'], temp.loc[i, 'sLon'], midLat, midLon) ** 2 * temp.loc[
                i, 'DuringTime']
    spentTime = list(temp['DuringTime'])
    rg = sqrt(sumdist / len(temp) / sum(spentTime))
    return rg

"""将轨迹按照日期切分"""
def get_file(temp):

    length = len(temp)
    temp['time_day'] = 0 * length
    for i in range(1, length + 1):
        temp.loc[i, 'time_day'] = temp.loc[i, 'Time'].split(' ', 2)[0]
    day_list = []
    # 获取日期列表
    for i in range(1, length + 1):
        day_list.append(temp.loc[i, 'time_day'])
    day_list = list(set(day_list))
    day_list = sorted(day_list)
    day_length = len(day_list)
    data_list = []
    llll = []
    for i in range(day_length):
        data_list.append(copy.deepcopy(llll))
    for i in range(1, length + 1):
        BeginIndex = day_list.index(temp.loc[i, 'time_day'])
        data_list[BeginIndex].append(temp.loc[i, 'labels'])
    travel_length = len(data_list)  # 模式总数量
    file = []
    for i in range(len(data_list)):
        s = str(data_list[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '')  # 去除单引号，逗号，每行末尾追加换行符
        file.append(s)
    return file, travel_length

def getLoPropo(temp):

    file,travel_length = get_file(temp)
    S_Loc = PrefixSpan.read(file)
    patterns = []
    patterns_Loc = PrefixSpan.prefixSpan(PrefixSpan.SquencePattern([], sys.maxsize), S_Loc, 2)
    #PrefixSpan.print_patterns(patterns_Loc)
    patterns_list,max_support = PrefixSpan.get_maxPatterns(patterns_Loc)
    del S_Loc
    #PrefixSpan.print_patterns(patterns_list)
    return patterns_list,max_support

def get_propo(path,filename):

    file = os.path.join(path,filename)
    temp = pd.read_csv(file,index_col=0)
    patterns, max_support = getLoPropo(temp)
    # PrefixSpan.print_patterns(patterns)
    PrefixSpan.print_patterns(patterns)
    def get_Day(temp):
        temp['Time'] = pd.to_datetime(temp['Time'])
        temp_six = temp[temp['Time'].dt.month == 6]
        temp_seven = temp[temp['Time'].dt.month == 7]
        temp_eight = temp[temp['Time'].dt.month == 8]
        temp_six['Time'] = pd.to_datetime(temp_six['Time'])
        temp_seven['Time'] = pd.to_datetime(temp_seven['Time'])
        temp_eight['Time'] = pd.to_datetime(temp_eight['Time'])
        day_len = len(set(temp_six['Time'].dt.day)) + \
                  len(set(temp_seven['Time'].dt.day)) + len(set(temp_eight['Time'].dt.day))
        return day_len
    day_len = get_Day(temp)
    propo = max_support / day_len * 100
    return propo

def get_move(path,filename):
    """获取移动特征"""
    #file = os.path.join(path,str(filename)+'.csv')
    file = os.path.join(path, filename)
    print("processing:",file)
    temp = pd.read_csv(file,index_col=0)
    day_len = get_day(temp)
    label_list = list(temp['Slabel'])+list(temp['Elabel'])
    ntrip = len(temp)
    aver_ntrip = ntrip//day_len
    aver_dtrip = sum(temp['Distance'])/ntrip
    aver_ttrip = sum(temp['TravelTime'])/ntrip
    ent_time   = get_entropy(temp)
    n_cluster = len(set(label_list)) - 1
    return aver_ntrip,aver_dtrip,aver_ttrip,ent_time,n_cluster

def get_stop(path,filename):
    """获取停等特征"""
    #file = os.path.join(path, str(filename) + '.csv')
    file = os.path.join(path,filename)
    print("processing:", file)
    temp = pd.read_csv(file, index_col=0)
    n_point = len(temp)
    aver_tstop = sum(temp['DuringTime'])/n_point
    #rg = get_rg(temp)
    return aver_tstop

def ana_Attri(path_in,filename):

    file = os.path.join(path_in,filename)
    print("processing:",file)
    temp = pd.read_csv(file,index_col=0)
    index_list = temp.columns.values.tolist()
    #if 'Slabel' in index_list:
    temp['StartTime'] = pd.to_datetime(temp['StartTime'])
    label_list = list(temp['Slabel'])+list(temp['Elabel'])
    #label_list2 = list(temp['STlabel'])+list(temp['ETlabel'])
    temp_six = temp[temp['StartTime'].dt.month==6]
    temp_seven = temp[temp['StartTime'].dt.month==7]
    temp_eight = temp[temp['StartTime'].dt.month==8]
    temp_six['StartTime']   = pd.to_datetime(temp_six['StartTime'])
    temp_seven['StartTime'] = pd.to_datetime(temp_seven['StartTime'])
    temp_eight['StartTime'] = pd.to_datetime(temp_eight['StartTime'])
        # print(set(temp_six['StartTime'].dt.day))
        #         # print(set(temp_seven['StartTime'].dt.day))
        #         # print(set(temp_eight['StartTime'].dt.day))
    #天数
    day_len = len(set(temp_six['StartTime'].dt.day))+\
                  len(set(temp_seven['StartTime'].dt.day))+len(set(temp_eight['StartTime'].dt.day))
    #空间簇数
    clu_num = len(set(label_list))-1
    #点总数
    point_num = len(temp)*2
    #总行程数
    trip_num = len(temp)
    #平均行程数
    avertrip = trip_num//day_len
    #起点是-1的行程
    acc_tripO = len(temp[temp['Slabel']==-1])
    #终点是-1的行程
    acc_tripD = len(temp[temp['Elabel']==-1])
    #离心点数目
    acc_num = acc_tripO+acc_tripD
    #非离心点行程
    no_acc_num = 0
    #时间簇数
    #tclu_num = len(set(label_list2))-1

    for i in range(1,len(temp)+1):
        if(temp.loc[i,'Slabel']!= -1 and temp.loc[i,'Elabel']!=-1):
                no_acc_num = no_acc_num + 1
        #print(filename,day_len)
    #return day_len,clu_num,point_num,trip_num,avertrip,acc_tripO,acc_tripD,acc_num,no_acc_num,tclu_num
    return clu_num
    # else:
    #     des_file = os.path.join(PATH_ACC,filename)
    #     shutil.move(file,des_file)
    #     return 0

if __name__ == "__main__":

    file_out = os.path.join(PATH_OUT, '所有数据特征集合.csv')
    temp = pd.read_csv(file_out)
    for i in range(len(temp)):
        temp.loc[i,'ObjectID'] = temp.loc[i,'ObjectID'][:-4]
    temp = temp.set_index(pd.Index(np.array(range(len(temp))) + 1))
    temp.to_csv(file_out)


    """
    f = open(file_out, 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['ObjectID','aver_ntrip','aver_dtrip','aver_trip','ent_time', 'n_cluster',
                         'aver_tstop'])
    f.close()
    filenames1 = os.listdir(PATH_IN)
    filenames2 = os.listdir(PATH_MOVE)
    filenames3 = os.listdir(PATH_STOP)
    filenames = list(set(filenames1+filenames2+filenames3))
    filenames.sort(key=lambda x: int(x[:-4]))
    for filename in filenames:
        if filename in filenames1:
            if filename in filenames2:
                if filename in filenames3:
                    print("processing:",filename)
                    aver_ntrip, aver_dtrip, aver_ttrip, ent_time, n_cluster = get_move(PATH_MOVE,filename)
                    #propo = get_propo(PATH_IN,filename)
                    aver_tstop = get_stop(PATH_STOP,filename)
                    f1 = open(file_out, 'a', encoding='utf-8', newline='')
                    csv.writer(f1).writerow([filename,aver_ntrip, aver_dtrip, aver_ttrip,
                                             ent_time,n_cluster,aver_tstop])
                    f1.close()
    """

    # for filename in IR_objectid:
    #
    #     print("processing:", filename)
    #     aver_ntrip, aver_dtrip, aver_ttrip, ent_time, n_cluster = get_move(PATH_MOVE,filename)
    #     aver_tstop, rg = get_stop(PATH_STOP,filename)
    #     f2 = open(file_out, 'a', encoding='utf-8', newline='')
    #     csv.writer(f2).writerow([filename,aver_ntrip, aver_dtrip, aver_ttrip,
    #                              ent_time,n_cluster,aver_tstop,rg,0])
    #     f2.close()


    """
    # rclu_num = []
    # iclu_num = []
    # filenames = os.listdir(PATH_REGURLAR)
    # filenames.sort(key=lambda x: int(x[:-4]))
    # for filename in filenames:
    #     rclu_num.append(ana_Attri(PATH_REGURLAR,filename))
    # filenames1 = os.listdir(PATH_IREGURLAR)
    # filenames1.sort(key=lambda x: int(x[:-4]))
    # for filename1 in filenames1:
    #     iclu_num.append(ana_Attri(PATH_IREGURLAR,filename1))
    # rbin = np.arange(min(rclu_num),max(rclu_num),1)
    # ibin = np.arange(min(iclu_num),max(iclu_num),1)
    # plt.hist(rclu_num,bins=rbin,density=True,facecolor='b',edgecolor='k',alpha=0.8)
    # plt.hist(iclu_num,bins=ibin,density=True,facecolor='darkorange',edgecolor='k',alpha=0.2)
    # plt.show()
    # plt.close()

    #filenames = os.listdir(PATH_IN1)
    #filenames = ['108831.csv','123212.csv']
    filenames = ['108831.csv']
    filenames.sort(key=lambda x: int(x[:-4]))
    for filename in filenames:
        file = os.path.join(PATH_IN1,filename)
        temp = pd.read_csv(file,index_col=0)
        # temp = temp[temp['Slabel']!=temp['Elabel']]
        # temp['StartTime'] = pd.to_datetime(temp['StartTime'])
        # temp['Month'] = temp['StartTime'].dt.month
        # temp['Day'] = temp['StartTime'].dt.day
        # for i in range(0,2):
        #     lenth = len(temp[temp['labels']==i])
        #     print("标签为"+str(i)+"的点个数"+str(lenth)+"个")
        temp['Time'] = pd.to_datetime(temp['Time'])
        temp['Month'] = temp['Time'].dt.month
        #temp = temp[temp['Month'] ==6]
        temp['Day'] = temp['Time'].dt.day
        temp['number'] = temp['Month']+temp['Day']/100
        temp = temp.head(16)

        #temp = temp.set_index(pd.Index(np.array(range(len(temp))) + 1))
        #print(temp)
        #print(temp.loc[:,['StartTime','Slabel','Elabel']])
        #x = temp['number']
        print(temp['Time'].dt.date)

        x = temp['Day']
        y = temp['Lon']
        z = temp['Lat']
        c = temp['labels']
        xticks = ['2016-06-01', '2016-06-02', '2016-06-03', '2016-06-04']
        print(len(set(c)))
        fig = plt.figure()
        ax = Axes3D(fig)
        # plt.scatter(y, z, c=c, cmap=plt.cm.Spectral)
        # plt.plot(y,z,c='r')
        #ax.view_init(elev=30, azim=90)
        ax.scatter(x,y,z,c=c,cmap=plt.cm.Spectral)
        #ax.scatter(x, y, z)
        ax.plot(x,y,z,c='r')
        #plt.show()
        plt.xticks([1,2,3,4],xticks,rotation=75)

        plt.title(str(filename[:-4]))
        plt.show()
        PATH_OUT = PATH_FIG
        #plt.savefig(os.path.join(PATH_OUT,'图5-2(b).png'))
        plt.close()

    file_out = os.path.join(PATH_OUT,'深圳市所有车辆统计表_工作日2.csv')
    temp = pd.read_csv(file_out)
    temp = temp.head(120)
    ObjectId = temp['ObjectID']
    for i in range(len(ObjectId)):
        objectid = ObjectId[i]
        print(objectid)
        src_file = os.path.join(PATH_WORK,str(objectid)+'.csv')
        des_file = os.path.join(PATH_IN,str(objectid)+'.csv')
        shutil.copy(src_file,des_file)
    """


