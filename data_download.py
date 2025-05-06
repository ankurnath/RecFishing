import wget
import os
import shutil

url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2019/AIS_2019_01_01.zip'

# Define the base data directory
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

# Local paths
zip_path = os.path.join(output_dir, os.path.basename(url))
extract_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(url))[0])

# Download if missing
if os.path.exists(zip_path):
    print(f"File already exists, skipping download: {zip_path}")
else:
    print(f"Downloading {url} …")
    wget.download(url, out=zip_path)
    print(f"\nDownloaded to {zip_path}")

# Unzip (skip if already extracted)
if os.path.isdir(extract_dir) and os.listdir(extract_dir):
    print(f"Archive already extracted to: {extract_dir}")
else:
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {zip_path} → {extract_dir} …")
    shutil.unpack_archive(zip_path, extract_dir)
    print("Extraction complete.")
