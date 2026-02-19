"""
Data aggregator to dowload and aggregate multitude of TC data-tracks
"""

import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from .download import download_storm_forecast, extract_gzip



# NHC URLs 
ARCHIVE_URL = "https://ftp.nhc.noaa.gov/atcf/archive/"

def get_stormfilenames(year, basin: str = "al", deck: str = "B"):
    """
    Gets the filenames for all stormfiles in a given year, basin, and deck.
    
    Args:
        year (int): Year to fetch storm data for
        basin (str): Basin identifier (e.g., 'al' for Atlantic, 'ep' for East Pacific)
        deck (str): Deck type - 'B' for best track (A deck not supported yet)
    
    Returns:
        list: List of storm filenames (e.g., ['bal012020.dat.gz', 'bal022020.dat.gz', ...])
    """
    if deck == "A":
        raise ValueError("Deck A is not supported yet. Please use Deck B.")
    elif deck != "B":
        raise ValueError("Invalid deck specified. Please use Deck B. (A deck is not supported yet)")
    
    url = f"{ARCHIVE_URL}{year}/"
    deck_lower = deck.lower()
    
    try:
        # Fetch the directory listing
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the HTML to extract filenames
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links that match the pattern: {deck}{basin}XX{year}.dat.gz
        # where XX is the storm number (01, 02, etc.)
        storm_files = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.startswith(f"{deck_lower}{basin}") and href.endswith(f"{year}.dat.gz"):
                storm_files.append(href)
        
        storm_files.sort()  # Sort the files for consistent ordering
        return storm_files
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching storm filenames from {url}: {e}")
        return []



def aggregate_tc_data(years: list, basin: str, deck: str = "B") -> None:
    """
    Downloads and extracts all B-deck track files for the given years and basin.

    Args:
        years: List of years to aggregate data for.
        basin: Basin identifier (e.g., 'al' for Atlantic).
        deck: Deck type — only 'B' is supported.
    """
    if deck == "A":
        raise ValueError("Deck A is not supported yet. Please use Deck B.")
    elif deck != "B":
        raise ValueError("Invalid deck specified. Please use Deck B.")

    # Collect all files across all years first so we know the total
    print(f"Fetching file lists for {len(years)} year(s)...")
    all_files = []  # list of (year, filename)
    for year in years:
        storm_files = get_stormfilenames(year, basin, deck)
        if not storm_files:
            print(f"  No files found for {year} ({basin.upper()}), skipping.")
        else:
            all_files.extend((year, f) for f in storm_files)

    if not all_files:
        print("Nothing to download.")
        return

    print(f"Found {len(all_files)} file(s) in total. Starting download...")
    failed = []
    for year, filename in tqdm(all_files, desc="Downloading", unit="file"):
        local_path = download_storm_forecast(year, basin, storm_id=filename[3:5], deck=deck, filename=filename, verbose=False)
        if local_path:
            extract_gzip(local_path, verbose=False)
            if os.path.exists(local_path):
                os.remove(local_path)
        else:
            failed.append(filename)

    if failed:
        print(f"\n{len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"  {f}")
    else:
        print("\nAll files downloaded successfully.")

    

    

