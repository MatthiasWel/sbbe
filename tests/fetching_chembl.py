from pathlib import Path

from chemford.fetch.chembl import fetch_activities_chembl, fetch_metadata_chembl

PATH_TO_CHEMBL = Path('/data/shared/exchange/uashehab/chembl35')

def fetch_activities():
    activities = fetch_activities_chembl(PATH_TO_CHEMBL)