import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitter import Fitter

PATH_IN  = "F:/大论文/实验/Data"
PATH_OUT = "F:/大论文/实验/Data"

"""
distribution = ['beta','burr', 'expon', 'exponnorm', 'exponpow', 'exponweib','gamma',
                 'genexpon', 'genextreme','genhalflogistic', 'genlogistic', 'gennorm',
                'genpareto', 'halfgennorm', 'halflogistic', 'halfnorm','invgamma', 'invgauss', 'invweibull',
                'truncexpon', 'truncnorm', 'uniform',  'weibull_max', 'weibull_min']
"""

distribution = [ 'expon', 'exponnorm', 'exponweib',
                'genpareto','invgamma', 'invgauss']

def fitterData(data,bin_num):
    f = Fitter(data)
    f.distributions = distribution
    f.bins = bin_num
    f.fit()
    f.summary(Nbest=1)
    print(f.get_best())
    #f.plot_pdf()
    #plt.semilogx()
    #plt.semilogy()
    #plt.show()

if __name__ =="__main__":

    filename = 'sz_rest.csv'
    file = os.path.join(PATH_IN,filename)
    temp = pd.read_csv(file,index_col=0)
    # data_traveltime = list(temp[temp['TravelTime']<=1000].TravelTime)
    data_distance = list(temp[temp['Distance']<=1000].Distance)
    #data_duringtime = list(temp[temp['DuringTime']<=1000].DuringTime)
    # data_rg = temp['Rg']
    x = data_distance
    bin_num = math.ceil(math.sqrt(len(x)))  #获取分组数
    print(bin_num)
    #n,bins,patches = plt.hist(x,bins=bin_num, facecolor='yellowgreen', edgecolor='purple', density=True)
    #fitterData(x,bin_num)
    #plt.xlim(0,200)
    # plt.close()
    # scatter_x = list(bins)
    # del scatter_x[0]
    # scatter_y = n
    # plt.scatter(scatter_x,scatter_y)
    #plt.show()

    #print(len(data))
    #print(data)
    #fitterData(x)
    print(1/1204.7843613256005/0.007876125880750585)