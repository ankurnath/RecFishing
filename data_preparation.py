import os
import wget
import shutil
import pandas as pd
from shapely.geometry import Point, Polygon
from datetime import date, timedelta
from time import time
from shapely import points, contains
from shapely.prepared import prep


# --- CONFIG ---
years    = [2019]
base_url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip'
output_dir = 'data'
zip_dir   = os.path.join(output_dir, 'zip')
csv_dir   = os.path.join(output_dir, 'csv')

os.makedirs(zip_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Load your Gulf polygon once:
polygon = pd.read_csv('Gulf_polygon.csv')
coords = list(zip(polygon['LAT'],polygon['LON']))
polygon = Polygon(coords)
prep_poly = prep(polygon)

def preprocess_csv(file_path):

    # Read only the needed columns
    df = pd.read_csv(
        file_path,
        usecols=['MMSI','VesselType','BaseDateTime','LAT','LON']
    ).dropna()



    lats = df['LAT'].values
    lons = df['LON'].values

    

    pt_geoms = points(lats, lons)
    
    inside_mask = contains(polygon, pt_geoms)
    

    df = df[inside_mask]

    

    # Compute elapsed seconds since first record of each MMSI
    df['dt'] = pd.to_datetime(df['BaseDateTime'])
    df['tod'] = df['dt'] - df['dt'].dt.normalize()
    df['elapsed_s'] = (
        df.groupby('MMSI')['tod']
          .transform(lambda x: (x - x.min()).dt.total_seconds().astype(int))
    )

    # Collapse into one row per MMSI
    df_coll = (
        df.groupby('MMSI', sort=False)
          .agg({
              'elapsed_s': list,
              'LAT':       list,
              'LON':       list,
              'VesselType':'first'
          })
          .reset_index()
    )
    # Create binary label and drop VesselType
    df_coll['Label'] = (df_coll['VesselType'] == 37).astype(int)
    # df_coll.drop(columns=['MMSI','VesselType'], inplace=True)

    # print(df_coll.head)

    return df_coll


save_folder = '../AIS_data/'
os.makedirs(save_folder, exist_ok=True)

start_time = time()
# --- MAIN LOOP ---
for year in years:
    start = date(year, 1, 1)
    end   = date(year + 1, 1, 1)
    current = start

    while current < end:
        yyyy, mm, dd = current.year, current.month, current.day

        zip_name = f'AIS_{yyyy}_{mm:02d}_{dd:02d}.zip'
        csv_name = zip_name.replace('.zip', '.csv')
        zip_path = os.path.join(zip_dir, zip_name)
        csv_path = os.path.join(csv_dir, csv_name)

        # 1) Download if missing
        if not os.path.exists(zip_path):
            url = base_url.format(year=yyyy, month=mm, day=dd)
            try:
                print(f"Downloading {zip_name} …")
                wget.download(url, out=zip_path)
                print()
            except Exception as e:
                print(f"  → failed: {e}")
                current += timedelta(days=1)
                continue  # skip to next day

        # 2) Extract if that day's CSV isn't already there
        if not os.path.exists(csv_path):
            try:
                print(f"Extracting {zip_name} …")
                shutil.unpack_archive(zip_path, csv_dir)
                os.remove(zip_path)
                print(f"  → extracted and removed zip")
            except Exception as e:
                print(f"  → extract failed: {e}")
                current += timedelta(days=1)
                continue

        # 3) Preprocess & save
        print(f"Preprocessing {csv_name} …")
        df_proc = preprocess_csv(csv_path)

        # print(df_proc)
        # print(df_proc.head())

        # Save to CSV
        save_path = os.path.join(save_folder, f'{yyyy}_{mm:02d}_{dd:02d}.pkl')
        df_proc.to_pickle(save_path)  # e.g., 'data.pkl'

        print(f"  → saved to {save_path}") 


        os.remove(csv_path)
        

        current += timedelta(days=1)

        # break

end_time = time()
print(f"Preprocessing took {end_time - start_time:.2f} seconds")


### --- TESTING ---
# df = pd.read_csv(save_path)

# print(df.head())