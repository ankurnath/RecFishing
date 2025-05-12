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

    for file in files:
        if file.endswith('.pkl') and file.startswith(tuple(years)):
            df = pd.read_pickle(os.path.join(folder, file))
            df = df[df['LAT'].apply(lambda x: len(x) >= min_data_points)]
            all_df = pd.concat([all_df, df], ignore_index=True)
            # break
    
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

    # # Map weekday string to integer (0 = Monday, 6 = Sunday)
    # weekday_map = {
    #     'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    #     'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    # }
    # weekday_num = weekday_map.get(row['weekday'], -1)  # default -1 if unknown

    return pd.Series({
        'duration': duration,
        'lat_mean': lat.mean(),
        'lat_std': lat.std(),
        'lon_mean': lon.mean(),
        'lon_std': lon.std(),
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
        # 'weekday': weekday_num,
    })
