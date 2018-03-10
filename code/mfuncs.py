import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

    
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

def add_dist_to_neighbours(df):
	df_point_dup = df.groupby(['pos_lat', 'pos_lon']).agg('size').reset_index()
	df_point_dup.columns = ['pos_lat', 'pos_lon', 'pos_customer_freq']
	df = pd.merge(df, df_point_dup, on=['pos_lat', 'pos_lon'], how='left')

	# расстояния до двух ближайщих соседей
	points_pos = df[['pos_lat', 'pos_lon']].dropna().values
	if points_pos.size:
		neigh_pos = NearestNeighbors(2)
		neigh_pos.fit(np.unique(points_pos, axis=1))  
	else:
		neigh_pos = None
	df_ = df.apply(lambda x: nearest_dist(x, neigh_pos, 'pos', 'pos2pos'), axis=1)
	df = pd.concat([df, df_], axis=1)
	df_ = df.apply(lambda x: nearest_dist(x, neigh_pos, 'atm', 'atm2pos'), axis=1)
	df = pd.concat([df, df_], axis=1)

	points_atm = df[['atm_lat', 'atm_lon']].dropna().values
	if points_atm.size:
		neigh_atm = NearestNeighbors(2)
		neigh_atm.fit(np.unique(points_atm, axis=1))  
	else:
		neigh_atm = None
	df_ = df.apply(lambda x: nearest_dist(x, neigh_atm, 'pos', 'pos2atm'), axis=1)
	df = pd.concat([df, df_], axis=1)
	df_ = df.apply(lambda x: nearest_dist(x, neigh_atm, 'atm', 'pos2atm'), axis=1)
	df = pd.concat([df, df_], axis=1)

	# neigh_all = NearestNeighbors(2)
	# neigh_all.fit(np.unique(np.unique(np.vstack([points_pos, points_atm]), axis=1), axis=1))


	return df


def nearest_dist(row, neigh, ttype, prefix):
    lat, lon = row[['{}_lat'.format(ttype), '{}_lon'.format(ttype)]].values
    if np.any(np.isnan([lat, lon])):
        distances = [-1, -1]
    elif not neigh:
    	distances = [-1, -1]
    else:
        distances, indices = neigh.kneighbors([[lat, lon]])
    return pd.Series(data=distances[0], index=['{}_1'.format(prefix), '{}_2'.format(prefix)])

def check_submit(path_to_csv):
    """
    Dummy checking of submission
    
    :param path_to_csv: path to your submission file
    """
    df = pd.read_csv(path_to_csv)
    assert df.shape == (9997, 5), u'Мало или много строк'
    # несмотря на то, что названия не имеют особого значения, правильный порядк колонок позволит не запутаться в широте-долготе
    assert list(df.columns) == ['_ID_', '_WORK_LAT_', '_WORK_LON_', '_HOME_LAT_', '_HOME_LON_'], u'Неверные названия столбцов'
    assert np.any(df['_ID_'].duplicated()) == False, u'Одному клиенту соответствует больше одной записи'
    for col_name in df.columns:
        if col_name != '_ID_':
            assert df[col_name].dtype in (np.float, np.int), u'В колонке {col_name} есть NULL'.format(col_name=col_name)
        assert df[col_name].isnull().sum() == 0, u'В колонке {col_name} есть NULL'.format(col_name=col_name)