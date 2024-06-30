# -*- coding: utf-8 -*-
import numpy as np
import csv
import math
import os
import shutil
from math import asin, cos, sin, sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PATH_FIG = r"C:\Users\58393\Desktop\H-paper\exp\Figure"
PATH_OUT  = r"C:\Users\58393\Desktop\H-paper\exp"

# file_out = os.path.join(PATH_OUT, '深圳市车辆模式比例.csv')
# temp = pd.read_csv(file_out)
# len1 = len(temp)
# temp = temp[temp['比例']<70]
# temp = temp[temp['label']==1]
# len2 = len(temp)
# print(len2/len1)

#coords = [temp['比例']]

# coords = [temp['aver_ntrip'], temp['aver_dtrip'],temp['aver_trip'],temp['ent_time']
#               ,temp['n_cluster'],temp['aver_tstop']]
# data = (np.array(coords))
# feature = np.transpose(data)
# clf = KMeans(n_clusters=2)
# s = clf.fit(feature)
# temp['label'] = s.labels_
# temp.to_csv(file_out)
file_out = os.path.join(PATH_OUT, '所有数据特征集合.csv')
temp = pd.read_csv(file_out,index_col=0)

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
#['ObjectID', 'aver_ntrip', 'aver_dtrip', 'aver_trip', 'ent_time', 'n_cluster', 'aver_tstop', 'label']
re_ntrip = []
ir_ntrip = []
for i in range(1,len(temp)+1):
    for j1 in Re_ObjectId:
        if(temp.loc[i,'ObjectID']==j1):
            re_ntrip.append(temp.loc[i,'aver_tstop']/60)
    for j2 in IR_objectid:
        if (temp.loc[i, 'ObjectID'] == j2):
            ir_ntrip.append(temp.loc[i, 'aver_tstop']/60)
rbin = np.arange(min(re_ntrip),max(re_ntrip),1)
ibin = np.arange(min(ir_ntrip),max(ir_ntrip),1)
print(rbin)
plt.hist(re_ntrip,bins=rbin,density=True,facecolor='b',edgecolor='k',alpha=0.8,label='regurlar')
plt.hist(ir_ntrip,bins=ibin,density=True,facecolor='darkorange',edgecolor='k',alpha=0.2,label='irregular')
plt.xlabel("aver_tstop(hour)")
plt.ylabel("pdf")
plt.legend()
#plt.show()
plt.savefig(os.path.join(PATH_FIG,'图5-4(aver_tstop).png'))
plt.show()
plt.close()
