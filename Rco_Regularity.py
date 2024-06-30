import os
import pandas as pd
import numpy as np
import math
import time
import datetime
import sys
import copy
import csv
import PrefixSpan

PATH_IN = r"your_path_in"
PATH_OUT = r"your_path_out"

def get_file(temp):

    length = len(temp)
    temp['time_day'] = 0 * length
    for i in range(1, length + 1):
        temp.loc[i, 'time_day'] = temp.loc[i, 'Time'].split(' ', 2)[0]
    day_list = []
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
    travel_length = len(data_list)  # Total number of patterns
    file = []
    for i in range(len(data_list)):
        s = str(data_list[i]).replace('[', '').replace(']', '')  # delete [],
        s = s.replace("'", '').replace(',', '')
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

def get_Day(temp):
    temp['Time'] = pd.to_datetime(temp['Time'])
    temp_six = temp[temp['Time'].dt.month==6]
    temp_seven = temp[temp['Time'].dt.month==7]
    temp_eight = temp[temp['Time'].dt.month==8]
    temp_six['Time']   = pd.to_datetime(temp_six['Time'])
    temp_seven['Time'] = pd.to_datetime(temp_seven['Time'])
    temp_eight['Time'] = pd.to_datetime(temp_eight['Time'])
    day_len = len(set(temp_six['Time'].dt.day)) + \
              len(set(temp_seven['Time'].dt.day)) + len(set(temp_eight['Time'].dt.day))
    return day_len

if __name__ == "__main__":

    # Time Lon Lat labels time Tlabels
    #117935.csv,121266.csv,128209.csv,184799.csv,402548.csv,533760.csv,551239.csv
    #551241.csv,569697.csv,569697.csv
    file_out = os.path.join(PATH_OUT,'shenzhen.csv')

    """
    f = open(file_out, 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['number','percentage'])
    f.close()
    """
    filenames = os.listdir(PATH_IN)
    filenames.sort(key=lambda x: int(x[:-4]))
    for filename in filenames:
        if(int(filename[:-4])==100605335):
            file = os.path.join(PATH_IN,filename)
            print("processing:", file)
            temp = pd.read_csv(file,index_col=0)
            patterns,max_support = getLoPropo(temp)
            #PrefixSpan.print_patterns(patterns)
            PrefixSpan.print_patterns(patterns)
            day_len = get_Day(temp)
            propo = max_support/day_len*100
            print(day_len,propo)
            # f1 = open(file_out, 'a', encoding='utf-8', newline='')
            # csv.writer(f1).writerow([filename[:-4],propo])
            # f1.close()
