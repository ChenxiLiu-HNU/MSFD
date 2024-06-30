import folium
import pandas as pd
import os
import numpy as np
import seaborn as sns
from folium.plugins import HeatMap

PATH = r"\exp\data"
file = os.path.join(PATH,"20160678.csv")
temp = pd.read_csv(file)
print(temp.columns.values.tolist())
lat = np.array(temp['StartLat'])
lon = np.array(temp['StartLon'])
new_Data = [[lat[i], lon[i]] for i in range(len(temp))]

world_map = folium.Map()

latitude = 37.31833
longitude = 110.485935

# Create map and display it
san_map = folium.Map(location=[lat.mean(),lon.mean()],
                     zoom_start=12,
                     tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                     attr="http://ditu.amap.com/")

HeatMap(new_Data,radius=20).add_to(san_map)

san_map.save('map.html')
