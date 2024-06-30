import pymysql
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import math
from math import asin, cos, sin, sqrt
import os
import csv

work_days5 = [1,4,5,6,7,8,11,12,13,14,15,18,19,20,21,22,25,26,27,28,29]
work_days6 = [1,2,4,5,6,7,8,9,11,12,13,14,15,16,18,19,20,21,22,23,25,26,27,28,29,30]
week_days2 = [2,3,9,10,16,17,23,24,30,31]
week_day = [3,10,17,24,31]


PATH_IN  = r"C:\Users\58393\Desktop\H-paper\exp\Data"
PATH_OUT = r"C:\Users\58393\Desktop\H-paper\exp"


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

def daoDataBase(base_name,table_name):
    engine = create_engine("mysql+mysqlconnector://root:123456@localhost:3306/"+base_name, encoding='utf-8')
    sql = "select * from" +" " +table_name
    return engine,sql

"""预处理操作:添加距离、时长"""
def propress_dt(filename):
    file = os.path.join(PATH_IN,filename)
    print(file)
    data = pd.read_csv(file,index_col=0)
    data['StartTime']= pd.to_datetime(data['StartTime'])
    data['StopTime'] = pd.to_datetime(data['StopTime'])
    for i in range(1,len(data)+1):

        day = float((data.loc[i, 'TravelTime'].split(" days ", 1))[0])
        time = (data.loc[i,'TravelTime'].split(" days ", 1))[1]
        hour = float((time.split(":", 2))[0])
        min = float((time.split(":", 2))[1])
        seconds = float((time.split(":", 2))[2])
        SpentTime = day * 24 * 60 + hour * 60 + min + seconds / 60
        data.loc[i,'Distance'] = distance(data.loc[i,'StartLat'],
                                             data.loc[i,'StartLon'],
                                             data.loc[i,'StopLat'],
                                             data.loc[i,'StopLon'])
        data.loc[i, 'TravelTime'] = SpentTime
        print("第" + str(i) + "行: ", data.loc[i, 'TravelTime'], data.loc[i, 'Distance'])
        #print("第" + str(i) + "行: ",data.loc[i,'TravelTime'])
    #outfile = os.path.join(PATH_OUT,outfile)
    #outfile = os.path.join(PATH_IN,filename)
    print(data)
    data.to_csv(file)

"""预处理操作：按id拆分表格"""
def propress_id(path_in,file,path_out):

    filename = os.path.join(path_in,file)
    print("process: ",filename)
    df = pd.read_csv(filename)
    print(len(set(df['ObjectID'])))
    for i in df.groupby('ObjectID'):
        temp = i[1].set_index(pd.Index(np.array(range(len(i[1]))) + 1))
        table_name = str(temp.loc[1, 'ObjectID'])+'.csv'
        pathout = path_out
        if not os.path.exists(pathout):
            os.makedirs(pathout)
        outname = os.path.join(pathout,table_name)
        print("outname: ",outname)
        temp.to_csv(outname)
        del temp
    print(".................finished.................")

def propress_merge(path_in,path_out,outname):
    filenames = os.listdir(path_in)
    file0 = os.path.join(path_in,filenames[0])
    temp0 = pd.read_csv(file0,index_col=0)
    for filename in filenames[1:]:
        file = os.path.join(path_in,filename)
        print("processing:",file)
        temp = pd.read_csv(file,index_col=0)
        temp0= pd.concat([temp0,temp])
    file_out = os.path.join(path_out,outname)
    temp0 = temp0.set_index(pd.Index(np.array(range(len(temp0))) + 1))
    temp0.to_csv(file_out)
    print(temp0)
    print("------------finished-------------")

def getMidPoint(temp):
    sumLon = sum(temp['sLon'])
    sumLat = sum(temp['sLat'])
    midLon = sumLon / (len(temp))
    midLat = sumLat / (len(temp))
    return midLon, midLat

def getRadius(path_in,filename):

    file = os.path.join(path_in,filename)
    temp = pd.read_csv(file,index_col=0)
    midLon,midLat = getMidPoint(temp)
    sumdist = 0
    for i in range(1, len(temp) + 1):
        sumdist = sumdist + distance(temp.loc[i, 'sLat'], temp.loc[i, 'sLon'], midLat, midLon) ** 2 * temp.loc[
                i, 'DuringTime']
        spentTime = list(temp['DuringTime'])
    rg = sqrt(sumdist / len(temp) / sum(spentTime))
    return rg

def getDay(temp):

    temp['StartTime'] = pd.to_datetime(temp['StartTime'])
    temp_six = temp[temp['StartTime'].dt.month==6]
    temp_seven = temp[temp['StartTime'].dt.month==7]
    temp_eight = temp[temp['StartTime'].dt.month==8]
    temp_six['StartTime']   = pd.to_datetime(temp_six['StartTime'])
    temp_seven['StartTime'] = pd.to_datetime(temp_seven['StartTime'])
    temp_eight['StartTime'] = pd.to_datetime(temp_eight['StartTime'])
    #天数
    day_len = len(set(temp_six['StartTime'].dt.day))+\
              len(set(temp_seven['StartTime'].dt.day))+len(set(temp_eight['StartTime'].dt.day))
    return(day_len)

"""
出行需求的分析
path_in:车辆工作日的行程文件夹
path_in:车辆周末的行程文件夹 
outname:输出图片名称
"""
def sta_traFre(path_in1,path_in2,outname):

    filenames1 = os.listdir(path_in1)
    filenames2 = os.listdir(path_in2)
    car_num1 = len(filenames1)
    car_num2 = len(filenames2)
    fre_work = []
    fre_week = []
    for filename1 in filenames1:
        print("filename1:",filename1)
        file1 = os.path.join(path_in1,filename1)
        temp1 = pd.read_csv(file1,index_col=0)
        days_num1 = getDay(temp1)
        fre = len(temp1)//days_num1
        print(fre)
        fre_work.append(fre)
    for filename2 in filenames2:
        print("filename2:", filename2)
        file2 = os.path.join(path_in2,filename2)
        temp2 = pd.read_csv(file2,index_col=0)
        days_num2 = getDay(temp2)
        fre = len(temp2)//days_num2
        print(fre)
        fre_week.append(fre)
    x1 = list(set(fre_work))
    x2 = list(set(fre_week))
    y1 = [0] * len(x1)
    y2 = [0] * len(x2)
    s1 = 0
    s2 = 0
    for i in x1:
        for j in range(len(fre_work)):
            if (fre_work[j] == i):
                y1[s1] = y1[s1] + 1
        s1 = s1 + 1

    for i in x2:
        for j in range(len(fre_week)):
            if (fre_week[j] == i):
                y2[s2] = y2[s2] + 1
        s2 = s2 + 1
    for i in range(len(y1)):
        y1[i] = y1[i] / len(fre_work) * 100
    for i in range(len(y2)):
        y2[i] = y2[i] / len(fre_week) * 100
    width_val = 0.4
    print(x1,y1)
    print(x2,y2)
    """
    for i in range(len(x2)):
         x2[i] = x2[i]+width_val
    plt.bar(x1, y1, alpha=0.6, width=width_val, facecolor='darkblue', edgecolor='darkblue', lw=1, label='weekdays')
    plt.bar(x2, y2, alpha=0.6, width=width_val, facecolor='deeppink', edgecolor='deeppink', lw=1, label='weekends')
    #plt.plot(x2,y2,color='deeppink',marker='o',markerfacecolor='deeppink',markersize=5,label='weekends')
    plt.xlabel("Travel Events Per vehicle Per Day")
    plt.ylabel("Probability of Travel Event Counts(%)")
    plt.xlim(0,30)
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig(os.path.join(PATH_OUT,outname))
    plt.close()
    """


"""
出行时刻的分析
filename1: 工作日行程文件
filename2: 周末行程文件
"""
def sta_time(filename1,filename2,outname):

    file1 = os.path.join(PATH_IN,filename1)
    file2 = os.path.join(PATH_IN,filename2)
    outfile = os.path.join(PATH_OUT,outname)

    data1 = pd.read_csv(file1)
    data1['StartTime'] = pd.to_datetime(data1['StartTime'])
    data1['Hour'] = data1['StartTime'].dt.hour
    data1_num = len(data1)

    data2 = pd.read_csv(file2)
    data2['StartTime'] = pd.to_datetime(data2['StartTime'])
    data2['Hour'] = data2['StartTime'].dt.hour
    data2_num = len(data2)

    """统计：这个时段的行程占总行程的比值"""
    outcome1 = [0]*24
    outcome2 = [0]*24
    time = np.arange(0,24,1)
    labels = ['0:00-1:00','1:00-2:00','2:00-3:00','3:00-4:00','4:00-5:00','5:00-6:00',
              '6:00-7:00','7:00-8:00','8:00-9:00','9:00-10:00','10:00-11:00','11:00-12:00',
              '12:00-13:00','13:00-14:00','14:00-15:00','15:00-16:00','16:00-17:00',
              '17:00-18:00','18:00-19:00','19:00-20:00','20:00-21:00','21:00-22:00','22:00-23:00',
              '23:00-24:00']
    for i in range(len(time)):
        outcome1[i] = len(data1[data1['Hour'] == time[i]])
        outcome2[i] = len(data2[data2['Hour'] == time[i]])

    for i in range(len(time)):
        outcome1[i] = outcome1[i]/data1_num*100
        outcome2[i] = outcome2[i]/data2_num*100

    print(outcome1)
    print(outcome2)

    """
    画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一(替换sans-serif字体)
    plt.rcParams['axes.unicode_minus'] = False
    #bar_width = 0.4
    # plt.bar(time,outcome1,bar_width,label='weekdays')
    # plt.bar(time+bar_width,outcome2,bar_width,label='weekends')
    plt.plot(time,outcome1,color='darkblue',marker='o',linewidth=2,markersize=5,label='weekdays')
    plt.plot(time,outcome2,color='goldenrod',marker='*',linewidth=2,markersize=5,label='weekends')
    plt.xlabel("Time of Hours")
    plt.ylabel("Proportion of Trips(%)")
    plt.xticks(time,labels,rotation=90)
    plt.legend()
    plt.grid()
    plt.savefig(outfile, bbox_inches='tight')
    #plt.show()
    plt.close()
    """

"""出行距离的分析"""
def sta_dist(filename,outname):

    file = os.path.join(PATH_IN,filename)
    data = pd.read_csv(file)
    data_num = len(data)
    distance = data[data['Distance']<100].Distance
    sum_dist = sum(distance)
    average_dist = sum_dist/data_num
    print(average_dist)
    width = 5
    bins1 = np.arange(0,max(distance),width)
    p1 = [0]*5
    p2 = [0]*5
    outfile = os.path.join(PATH_OUT,outname)
    n, bins, patches = plt.hist(distance, bins=bins1, facecolor='yellowgreen', edgecolor='k', density=True)
    plt.close()
    for i in range(4):
        p1[i] = n[i] * 5
    p1[4] = 1 - (n[0] + n[1] + n[2] + n[3]) * 5
    print(p1)


    """
    langs = ['<5km', '5-10km', '10-15km', '15-20km', '>20km']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一(替换sans-serif字体)
    plt.rcParams['axes.unicode_minus'] = False
    plt.pie(p1, labels=langs, autopct='%1.2f%%')
    plt.savefig(outfile)
    plt.close()
    """

"""出行时长的分析
path_in:文件路径
filename:文件名
"""
def sta_tradura(path_in,filename):

    file = os.path.join(path_in,filename)
    data = pd.read_csv(file,index_col=0)
    data = data[data['IsSunday']==1]
    data_valid = data[data['TravelTime']<300]
    data_valid = data_valid.set_index(pd.Index(np.array(range(len(data_valid))) + 1))
    duration = data_valid['TravelTime']
    sum_duration = sum(duration)
    data_num = len(data)
    average_duration = sum_duration/data_num
    """
    p = [0] * 5
    for i in range(1,len(duration)+1):
        if(duration[i]<15):
            p[0] = p[0]+1
        if(duration[i]>=15 and duration[i]<30):
            p[1] = p[1]+1
        if (duration[i]>=30 and duration[i]<45):
            p[2] = p[2] + 1
        if (duration[i]>=45 and duration[i]<60):
            p[3] = p[3] + 1
        if(duration[i]>=60):
            p[4] = p[4] + 1
    for i in range(len(p)):
        p[i] = p[i]/len(duration)
    """
    #print(filename,p)
    print(filename,average_duration)
    """
    langs = ['<15min', '15-30min', '30-45min', '45-60min', '>1hour']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一(替换sans-serif字体)
    plt.rcParams['axes.unicode_minus'] = False
    plt.pie(p, labels=langs, autopct='%1.2f%%')
    plt.savefig(os.path.join(PATH_OUT,outname))
    plt.show()
    plt.close()
    """

"""拥塞"""
def sta_conges(filename1,filename2):
    #filenames = os.listdir(PATH_IN)
    dist_times = {5:15,10:30,15:45,20:60}
    color = ['r','b','k','y']
   #for filename in filenames:
    file1 = os.path.join(PATH_IN,filename1)
    temp = pd.read_csv(file1, index_col=0)
    lenth = len(temp)
    temp['StartTime'] = pd.to_datetime(temp['StartTime'])
    temp['Hour'] = temp['StartTime'].dt.hour
    c = 0
    for dist,time in dist_times.items():
        temp1= temp[temp['Distance']<dist]
        temp2= temp1[temp1['Duration']>time]
        outcome = [0]*24
        time = np.arange(0,24,1)
        for i in time:
            temp_i = temp[temp['Hour']==i]
            temp_ab = temp2[temp2['Hour']==i]
            temp_a = temp1[temp1['Hour']==i]
            p1 = len(temp_ab)/len(temp_i)
            p2 = len(temp_a)/len(temp_i)
            outcome[i] = p1/p2*100
        plt.plot(time,outcome,c=color[c],marker='o',label='weekdays')
        c = c + 1
    file2 = os.path.join(PATH_IN, filename2)
    temp = pd.read_csv(file2, index_col=0)
    lenth = len(temp)
    temp['StartTime'] = pd.to_datetime(temp['StartTime'])
    temp['Hour'] = temp['StartTime'].dt.hour
    c = 0
    for dist, time in dist_times.items():
        temp1 = temp[temp['Distance'] < dist]
        temp2 = temp1[temp1['Duration'] > time]
        outcome = [0] * 24
        time = np.arange(0, 24, 1)
        for i in time:
            temp_i = temp[temp['Hour'] == i]
            temp_ab = temp2[temp2['Hour'] == i]
            temp_a = temp1[temp1['Hour'] == i]
            p1 = len(temp_ab) / len(temp_i)
            p2 = len(temp_a) / len(temp_i)
            outcome[i] = p1 / p2 * 100
        plt.plot(time, outcome, c=color[c],marker='+',label = 'weekend')
        c = c + 1
    plt.xticks(time,time)
    plt.grid()
    plt.savefig(os.path.join(PATH_OUT,filename1[:-11])+'_cogest.png')
    plt.close()


def getSDay(temp):

    temp['sTime'] = pd.to_datetime(temp['sTime'])
    temp_six = temp[temp['sTime'].dt.month==6]
    temp_seven = temp[temp['sTime'].dt.month==7]
    temp_eight = temp[temp['sTime'].dt.month==8]
    temp_six['sTime']   = pd.to_datetime(temp_six['sTime'])
    temp_seven['sTime'] = pd.to_datetime(temp_seven['sTime'])
    temp_eight['sTime'] = pd.to_datetime(temp_eight['sTime'])
    #天数
    day_len = len(set(temp_six['sTime'].dt.day))+\
              len(set(temp_seven['sTime'].dt.day))+len(set(temp_eight['sTime'].dt.day))
    return(day_len)

"""停等区域统计"""
def sta_staypoint(path_in1,path_in2):
    filenames1 = os.listdir(path_in1)
    filenames2 = os.listdir(path_in2)
    car_num1 = len(filenames1)
    car_num2 = len(filenames2)
    fre_work = []
    fre_week = []
    for filename1 in filenames1:
        print("filename1:", filename1)
        file1 = os.path.join(path_in1, filename1)
        temp1 = pd.read_csv(file1, index_col=0)
        days_num1 = getSDay(temp1)
        fre = len(temp1) // days_num1
        fre_work.append(fre)
    for filename2 in filenames2:
        print("filename2:", filename2)
        file2 = os.path.join(path_in2, filename2)
        temp2 = pd.read_csv(file2, index_col=0)
        days_num2 = getSDay(temp2)
        fre = len(temp2) // days_num2
        fre_week.append(fre)
    x1 = list(set(fre_work))
    x2 = list(set(fre_week))
    y1 = [0] * len(x1)
    y2 = [0] * len(x2)
    s1 = 0
    s2 = 0
    for i in x1:
        for j in range(len(fre_work)):
            if (fre_work[j] == i):
                y1[s1] = y1[s1] + 1
        s1 = s1 + 1

    for i in x2:
        for j in range(len(fre_week)):
            if (fre_week[j] == i):
                y2[s2] = y2[s2] + 1
        s2 = s2 + 1
    for i in range(len(y1)):
        y1[i] = y1[i] / len(fre_work) * 100
    for i in range(len(y2)):
        y2[i] = y2[i] / len(fre_week) * 100
    #width_val = 0.4
    print(x1, y1)
    print(x2, y2)

"""停等时刻"""
def sta_staytime(filename1,filename2):

    file1 = os.path.join(PATH_IN, filename1)
    file2 = os.path.join(PATH_IN, filename2)
    #outfile = os.path.join(PATH_OUT, outname)
    data1 = pd.read_csv(file1)
    data1['sTime'] = pd.to_datetime(data1['sTime'])
    data1['Hour'] = data1['sTime'].dt.hour
    data1_num = len(data1)

    data2 = pd.read_csv(file2)
    data2['sTime'] = pd.to_datetime(data2['sTime'])
    data2['Hour'] = data2['sTime'].dt.hour
    data2_num = len(data2)

    """统计：这个时段的行程占总行程的比值"""
    outcome1 = [0] * 24
    outcome2 = [0] * 24
    time = np.arange(0, 24, 1)
    labels = ['0:00-1:00', '1:00-2:00', '2:00-3:00', '3:00-4:00', '4:00-5:00', '5:00-6:00',
              '6:00-7:00', '7:00-8:00', '8:00-9:00', '9:00-10:00', '10:00-11:00', '11:00-12:00',
              '12:00-13:00', '13:00-14:00', '14:00-15:00', '15:00-16:00', '16:00-17:00',
              '17:00-18:00', '18:00-19:00', '19:00-20:00', '20:00-21:00', '21:00-22:00', '22:00-23:00',
              '23:00-24:00']
    for i in range(len(time)):
        outcome1[i] = len(data1[data1['Hour'] == time[i]])
        outcome2[i] = len(data2[data2['Hour'] == time[i]])

    for i in range(len(time)):
        outcome1[i] = outcome1[i] / data1_num * 100
        outcome2[i] = outcome2[i] / data2_num * 100

    print(outcome1)
    print(outcome2)

"""停等时长"""
def sta_staydura(path_in,filename):

    file = os.path.join(path_in,filename)
    data = pd.read_csv(file,index_col=0)
    duration = data.DuringTime
    sum_duration = sum(duration)
    data_num = len(data)
    average_duration = sum_duration / data_num / 60
    print("平均停等时长：",average_duration)
    #"""
    p = [0] * 4
    for i in range(1,len(duration)+1):
        if (duration[i] < 120):
            p[0] = p[0] + 1
        if (duration[i] >= 120 and duration[i] < 640):
            p[1] = p[1] + 1
        if (duration[i] >= 640 and duration[i] < 780):
            p[2] = p[2] + 1
        if (duration[i] >= 780):
            p[3] = p[3] + 1
    for i in range(len(p)):
        p[i] = p[i] / len(duration)
    print(file,p)
    #"""
    """
    duration = data[data['Duration'] < 300].Duration
    sum_duration = sum(duration)
    data_num = len(data)
    average_duration = sum_duration / data_num
    print(average_duration)

    p = [0] * 5
    for i in range(len(duration)):
        if(duration[i]<15):
            p[0] = p[0]+1
        if(duration[i]>=15 and duration[i]<30):
            p[1] = p[1]+1
        if (duration[i]>=30 and duration[i]<45):
            p[2] = p[2] + 1
        if (duration[i]>=45 and duration[i]<60):
            p[3] = p[3] + 1
        if(duration[i]>=60):
            p[4] = p[4] + 1
    for i in range(len(p)):
        p[i] = p[i]/len(duration)
    print(p)
    """
    # langs = ['<15min', '15-30min', '30-45min', '45-60min', '>1hour']
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一(替换sans-serif字体)
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.pie(p, labels=langs, autopct='%1.2f%%')
    # plt.savefig(os.path.join(PATH_OUT,outname))
    # plt.show()
    # plt.close()


if __name__ == "__main__":

    propress_id("E:/数据/GPS",'std07.csv',"E:/数据/gps_id07")
    #propress_id("F:/大论文/实验/Data",'sz_rest.csv',"F:/大论文/实验/Data/sz_rest")
    # propress_merge("F:/大论文/实验/Data/sz_work_stop","F:/大论文/实验/Data",'sz_work_stop.csv')
    # propress_merge("F:/大论文/实验/Data/sz_rest_stop","F:/大论文/实验/Data",'sz_rest_stop.csv')

    """出行频率"""
    #sta_traFre("F:/大论文/实验/Data/sz_work","F:/大论文/实验/Data/sz_rest",'Fig3.1.png')

    """出行时刻"""
    #sta_time("sz_work.csv","sz_rest.csv",'Fig3.2.png')

    """出行距离"""
    # sta_dist("sz_work.csv","Fig.3.4(a).png")
    # sta_dist('sz_rest.csv',"Fig.3.5(a).png")

    """出行时长"""
    # path_in = "F:/大论文/实验/Data/sz_time"
    # filenames = os.listdir(path_in)
    # for filename in filenames:
    #     sta_tradura(path_in,filename)
    #sta_tradura("F:/大论文/实验/Data","sz_weekend1.csv")

    """停等点分布"""
    #sta_staypoint("F:/大论文/实验/Data/sz_work_stop","F:/大论文/实验/Data/sz_rest_stop")

    """停等时刻"""
    #sta_staytime("F:/大论文/实验/Data/sz_work_stop.csv","F:/大论文/实验/Data/sz_rest_stop.csv")

    """停等时长"""
    # sta_staydura(PATH_IN,'sz_work_stop.csv')
    # sta_staydura(PATH_IN,'sz_rest_stop.csv')
    """
    PATH_WORK = "F:/大论文/实验/data/sz_work_stop"
    PATH_REST = "F:/大论文/实验/data/sz_rest_stop"
 
    filenames = os.listdir(PATH_WORK)
    filenames.sort(key=lambda x: int(x[:-4]))
    file_out1 = os.path.join(PATH_OUT, 'rg_work.csv')
    f = open(file_out1, 'a', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    # 3. 构建列表头
    csv_writer.writerow(["ObjectID","Rg"])
    f.close()
    for filename in filenames:
        print("processing:", filename)
        Rg = getRadius(PATH_WORK,filename)
        f1 = open(file_out1, 'a', encoding='utf-8', newline='')
        csv.writer(f1).writerow([filename[:-4],Rg])
        f1.close()
    print(".......................work out .............................")
    filenames2 = os.listdir(PATH_REST)
    filenames2.sort(key=lambda x: int(x[:-4]))
    file_out2 = os.path.join(PATH_OUT,'rg_rest.csv')
    f2 = open(file_out2, 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(f2)
    # 3. 构建列表头
    csv_writer.writerow(["ObjectID", "Rg"])
    f2.close()
    for filename in filenames2:
        print("processing:", filename)
        Rg = getRadius(PATH_REST, filename)
        f3 = open(file_out2, 'a', encoding='utf-8', newline='')
        csv.writer(f3).writerow([filename[:-4], Rg])
        f3.close()
    """


