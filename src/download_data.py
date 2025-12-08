import os
import requests
import gzip
import shutil
from tqdm import tqdm

# Constants
BASE_URL = "https://ftp.nhc.noaa.gov/atcf/archive/"
DATA_DIR = "./data/raw/"

def download_storm_forecast(year, basin, storm_id, deck ="b"):
    """
    Downloads the ATCF a/b-deck file for a specific storm.
    
    Args:
        year (int): Year of the storm (e.g., 2018)
        basin (str): 'al' for Atlantic, 'ep' for East Pacific
        storm_id (str): 2-digit ID (e.g., '14' for Michael)
    """

    filename = f"{deck}{basin}{storm_id}{year}.dat.gz"
    url = f"{BASE_URL}{year}/{filename}"
    
    local_path = os.path.join(DATA_DIR, filename)
    
    print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        print(f"Downloaded to {local_path}")
        return local_path
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        return None

def extract_gzip(file_path):
    """Decompresses the .gz file to a .dat file."""
    output_path = file_path.replace('.gz', '')
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted to {output_path}")
    return output_path

if __name__ == "__main__":
    # Example: Hurricane Michael (2018) - AL14
    # This was a major Cat 5 storm, great for testing loss models.
    os.makedirs(DATA_DIR, exist_ok=True)
    
    zip_path = download_storm_forecast(2018, "al", "14")
    
    if zip_path:
        extract_gzip(zip_path)
        os.remove(zip_path)

