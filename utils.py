import os
import pandas as pd
import joblib
import numpy as np

def load_filtered_data(years, folder='../AIS_data', min_data_points=100):
    """
    Load and filter AIS .pkl files from specified years.
    
    Args:
        years (tuple or list): Years to include, e.g., ('2022', '2023').
        folder (str): Folder containing .pkl files.
        min_data_points (int): Minimum required data points (length of LAT) in a trip.
    
    Returns:
        pd.DataFrame: Concatenated and filtered dataframe.
    """
    files = os.listdir(folder)
    all_df = pd.DataFrame()
    selected_MMSI = pd.read_csv('VT37MMSINumbers.csv')

    for file in files:
        if file.endswith('.pkl') and file.startswith(tuple(years)):
            df = pd.read_pickle(os.path.join(folder, file))
            df = df[df['LAT'].apply(lambda x: len(x) >= min_data_points)]
            # date_str = file.replace('.pkl', '')  # '2019_05_11'
            # date_obj = pd.to_datetime(date_str, format='%Y_%m_%d')
            # df['date'] = date_obj
            # df['weekday'] = date_obj.strftime('%A')  # e.g., 'Monday'

            all_df = pd.concat([all_df, df], ignore_index=True)

    all_df['Label'] = all_df['MMSI'].isin(selected_MMSI['MMSI']).astype(int)

    
    return all_df



# from sklearn.metrics import classification_report

def save_model(model, save_path='best_model.pkl'):
    """
     saves the model.

    """


    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")


def load_model(load_path='best_model.pkl'):
    """
    Loads a model from the given path.
    
    Args:
        load_path (str): Path to the saved model file.

    Returns:
        model: The loaded scikit-learn model.
    """
    model = joblib.load(load_path)
    print(f"Model loaded from {load_path}")
    return model

def extract_features(row):
    elapsed = np.array(row['elapsed_s'])
    lat = np.array(row['LAT'])
    lon = np.array(row['LON'])
    

    # Calculate duration
    duration = elapsed[-1] - elapsed[0]

    # Calculate simple speed approximations
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    dt = np.diff(elapsed) + 1e-6  # prevent division by zero
    speeds = np.sqrt(dlat**2 + dlon**2) / dt

    accel = np.diff(speeds) / (dt[1:] + 1e-6) if len(speeds) > 1 else np.array([0.0])

    return pd.Series({
        'duration': duration,
        'lat_mean': lat.mean(),
        'lon_mean': lon.mean(),
        'lat_std': lat.std(),
        'lon_std': lon.std(),
        'lat_median': np.median(lat),
        'lon_median': np.median(lon),
        'lat_min': lat.min(),
        'lon_min': lon.min(),
        'lat_max': lat.max(),
        'lon_max': lon.max(),
        'lat_range': lat.max() - lat.min(),
        'lon_range': lon.max() - lon.min(),
        'bbox_area': (lat.max() - lat.min()) * (lon.max() - lon.min()),
        'start_lat': lat[0],
        'start_lon': lon[0],
        'end_lat': lat[-1],
        'end_lon': lon[-1],
        'speed_mean': speeds.mean(),
        'speed_max': speeds.max(),
        'speed_std': speeds.std(),
        'accel_mean': accel.mean(),
        'accel_std': accel.std(),
        'accel_max': accel.max(),
        # 'wspd': row['WSPD'],
        # 'gst': row['GST'],
        # 'wvht': row['WVHT'],
        # 'atmp': row['ATMP']
        
    })


# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import impute
# from tsfresh.feature_extraction import EfficientFCParameters

# def extract_tsfresh_features(df):
#     """
#     Applies tsfresh to extract time-series features from trajectories.
#     Assumes input df has columns:
#         - 'id' : identifier per trajectory
#         - 'elapsed_s' : list of timestamps
#         - 'LAT' : list of latitudes
#         - 'LON' : list of longitudes
#         - optional: 'WSPD', 'GST', 'WVHT', 'ATMP'

#     Returns:
#         - DataFrame of tsfresh features joined with optional static features.
#     """
#     long_df = []

#     # for _, row in df.iterrows():
#     for id, row in df.iterrows():
#         traj_id = id
#         elapsed = row['elapsed_s']
#         lat = row['LAT']
#         lon = row['LON']

#         for t, (e, la, lo) in enumerate(zip(elapsed, lat, lon)):
#             long_df.append({
#                 'id': traj_id,
#                 'time': e,
#                 'lat': la,
#                 'lon': lo
#             })

#     long_df = pd.DataFrame(long_df)

#     # Compute speed and acceleration
#     def compute_speed_and_accel(group):
#         group = group.sort_values('time')
#         group['dlat'] = group['lat'].diff()
#         group['dlon'] = group['lon'].diff()
#         group['dt'] = group['time'].diff().replace(0, 1e-6)
#         group['speed'] = np.sqrt(group['dlat']**2 + group['dlon']**2) / group['dt']
#         group['accel'] = group['speed'].diff() / group['dt']
#         return group

#     long_df = long_df.groupby('id').apply(compute_speed_and_accel).reset_index(drop=True)

#     # print(long_df.head())
#     long_df.dropna(inplace=True)  # drop rows with NaN values

#     print(long_df.head())

#     # Extract features with tsfresh
#     features = extract_features(
#         long_df,
#         column_id="id",
#         column_sort="time",
#         default_fc_parameters=EfficientFCParameters(),
#         n_jobs=10
#         # impute_function=impute
#     )

#     return features



