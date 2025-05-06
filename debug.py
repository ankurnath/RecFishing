import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
from time import time



file_path = 'data/AIS_2019_01_01/AIS_2019_01_01.csv'

# poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])


# Read the CSV file

df = pd.read_csv(file_path)
df = df[['MMSI','VesselType','BaseDateTime','LAT', 'LON']]
df = df.dropna()

# print(df.head())


polygon = pd.read_csv('Gulf_polygon.csv')
coords = list(zip(polygon['LAT'],polygon['LON']))
polygon = Polygon(coords)

# bounding box
minx, miny, maxx, maxy = polygon.bounds

print(minx, miny, maxx, maxy)

lats = df['LAT'].values
lons = df['LON'].values

# boolean mask: only these points need the expensive test
bbox_mask = (lons >= minx) & (lons <= maxx) & (lats >= miny) & (lats <= maxy)

# indices of candidate points
idxs = np.nonzero(bbox_mask)[0]

print(df.iloc[idxs])



from shapely.prepared import prep
from shapely.geometry import Point

prep_poly = prep(polygon)

inside = np.zeros_like(lons, dtype=bool)

# start = time()
# for i in idxs:
#     pt = Point(lons[i], lats[i])
#     inside[i] = prep_poly.contains(pt)

# end = time()
# print("Time taken for polygon containment test:", end - start)
# print(sum(inside))

# from shapely.vectorized import contains

start = time()
# Vectorized operation
from shapely import points, contains

# build all Point geometries at once
# pt_geoms = points(lons, lats)        # returns a GeometryArray
pt_geoms = points(lats, lons)        # returns a GeometryArray

# test containment in one shot
inside_mask = contains(polygon, pt_geoms)

end = time()
print("Time taken for vectorized polygon containment test:", end - start)
print(sum(inside_mask))

# df = df.iloc[idxs][inside_mask]

# df = df.reset_index(drop=True)
# print(df.head())

# df = df[['MMSI','VesselType','BaseDateTime','LAT', 'LON']]

# df['dt'] = pd.to_datetime(df['BaseDateTime'])

# # print(df.columns)
# print(df.head())

# print(df['VesselType'].unique().tolist())
# print(df['VesselType'].value_counts())
# print(df[df['MMSI'] == 366876000])
# df['TimeStamp']= pd.to_datetime(df['BaseDateTime'], format='%Y-%m-%d %H:%M:%S')
# print("Number of points inside polygon:", sum(inside_mask))


