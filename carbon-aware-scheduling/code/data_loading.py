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
