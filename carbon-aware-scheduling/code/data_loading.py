"""
Step 1: download and load the three raw hourly sources (carbon intensity, weather, electricity prices).

Folder layout of the HPE data: <source>/<REGION>/<YEAR>/...
The folder name is used as the region id because it is consistent across all three sources
(the `Zone Id` column inside the CSVs is not, e.g. folder DE-LU has Zone Id DE).
"""
import glob
import json
import subprocess
from pathlib import Path
import pandas as pd
from settings import DATA_DIR, HPE_REPO_URL


def download_data(data_dir: Path = DATA_DIR) -> None:
    """
    Sparse-clone only the data/ folder of the HPE SustainCluster repository (~290 MB) if it is missing.
    :param data_dir: folder where the dataset is expected (data/sustain-cluster/data)
    :return: None
    """
    if data_dir.exists():
        print("Data already present:", data_dir)
        return
    dst = data_dir.parent
    print("Downloading HPE SustainCluster data to", dst, "...")
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", HPE_REPO_URL, str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "sparse-checkout", "set", "data"], check=True)


def list_regions(data_dir: Path = DATA_DIR) -> list:
    """
    Get the grid regions available in the dataset.
    :param data_dir: root folder of the dataset
    :return: sorted list of region ids, e.g. ["AU-NSW", "DE-LU", ...]
    """
    return sorted(p.name for p in (data_dir / "carbon_intensity").iterdir()
                  if p.is_dir() and not p.name.startswith("_"))


def load_carbon_intensity(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load the hourly carbon intensity files for all regions and years.
    :param data_dir: root folder of the dataset
    :return: raw carbon intensity rows with a `region` column
    """
    frames = []
    for f in sorted(glob.glob(str(data_dir / "carbon_intensity" / "*" / "*" / "*.csv"))):
        d = pd.read_csv(f)
        d["region"] = Path(f).parts[-3]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_weather(data_dir: Path = DATA_DIR):
    """
    Load the hourly weather JSON files for all regions.
    Weather timestamps are in local time, with one fixed UTC offset per file
    (`utc_offset_seconds`), so the offsets are returned for converting to UTC later.
    :param data_dir: root folder of the dataset
    :return: (weather rows, {region: timezone name}, {region: set of UTC offsets in seconds})
    """
    frames, region_tz, w_offset = [], {}, {}
    for f in sorted(glob.glob(str(data_dir / "weather" / "*" / "*.json"))):
        with open(f) as fh:
            j = json.load(fh)
        region = Path(f).parts[-2]
        d = pd.DataFrame(j["hourly"])
        d["region"] = region
        region_tz[region] = j["timezone"]
        w_offset.setdefault(region, set()).add(j["utc_offset_seconds"])
        frames.append(d)
    return pd.concat(frames, ignore_index=True), region_tz, w_offset

