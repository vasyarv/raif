import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from numpy.linalg import norm

def nearest_dist(row, neigh, ttype, prefix):
    lat, lon = row[['{}_lat'.format(ttype), '{}_lon'.format(ttype)]].values
    if np.any(np.isnan([lat, lon])):
        distances = [-1, -1]
    else:
        distances, indices = neigh.kneighbors([[lat, lon]])
    return pd.Series(data=distances[0], index=['{}_1'.format(prefix), '{}_2'.format(prefix)])
    
def rgb(value):
    minimum, maximum = float(0), float(255)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return (r, g, b)

def dist_latlon(lat1, lon1, lat2, lon2):
    return norm([lat1 - lat2, lon1 - lon2])

def add_poswork_target(x):
    lat1, lon1, lat2, lon2 = x[['pos_lat', 'pos_lon', 'work_lat', 'work_lon']]
    d = dist_latlon(lat1, lon1, lat2, lon2)
    return int(d < 0.02)

def add_poshome_target(x):
    lat1, lon1, lat2, lon2 = x[['pos_lat', 'pos_lon', 'home_lat', 'home_lon']]
    d = dist_latlon(lat1, lon1, lat2, lon2)
    return int(d < 0.02)