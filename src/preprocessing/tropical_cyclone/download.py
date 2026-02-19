import os
import requests
import gzip
import shutil
from datetime import datetime

# NHC URLs 
ARCHIVE_URL = "https://ftp.nhc.noaa.gov/atcf/archive/"
REALTIME_URL = "https://ftp.nhc.noaa.gov/atcf/aid_public/"
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKING_DIR,"../../../", "data", "raw")


# -------------------------------------------------------------------------
# SECTION: Download Functions
# -------------------------------------------------------------------------
def download_storm_forecast(year, basin, storm_id, deck="b", filename: str = None, verbose: bool = True):
    """
    Downloads the ATCF a/b-deck (or real time) file for a specific storm.
    
    Args:
        year (int): Year of the storm (e.g., 2018)
        basin (str): 'al' for Atlantic, 'ep' for East Pacific
        storm_id (str): 2-digit ID (e.g., '14' for Michael)
        filename (str, optional): Custom filename for the downloaded file. Defaults to None.
        verbose (bool): Print download status messages. Defaults to True.
    """
    current_year = datetime.now().year

    # Determine if we use archive or real-time URL
    if year == current_year:
        if verbose: print(f"Year is {year}. Checking REAL-TIME source...")
        base_url = REALTIME_URL
        if filename is None:
            filename = f"{deck}{basin}{storm_id}{year}.dat.gz"
        url = f"{base_url}{filename}"
    else:
        if verbose: print(f"Year is {year}. Checking ARCHIVE source...")
        base_url = ARCHIVE_URL
        if filename is None:
            filename = f"{deck}{basin}{storm_id}{year}.dat.gz"
        url = f"{base_url}{year}/{filename}"
    
    local_path = os.path.join(DATA_DIR, filename)
    
    # Check if file already exists
    if os.path.exists(local_path):
        if verbose: print(f"File already exists at {local_path}. Skipping download.")
        return local_path
    
    if verbose: print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose: print(f"Downloaded to {local_path}")
        return local_path
    else:
        if verbose: print(f"Failed to download. Status code: {response.status_code}")
        return None

def extract_gzip(file_path, verbose: bool = True):
    """Decompresses the .gz file to a .dat file."""
    output_path = file_path.replace('.gz', '')
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if verbose: print(f"Extracted to {output_path}")
    return output_path

if __name__ == "__main__":
    # Example: Hurricane Michael (2018) - AL14
    # This was a major Cat 5 storm, great for testing loss models.
    os.makedirs(DATA_DIR, exist_ok=True)
    
    zip_path = download_storm_forecast(2018, "al", "14")
    
    if zip_path:
        extract_gzip(zip_path)
        os.remove(zip_path)

