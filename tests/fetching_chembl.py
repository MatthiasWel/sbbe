from pathlib import Path
import pandas as pd
from chemford.fetch.chembl import fetch_activities_chembl

PATH_TO_CHEMBL = Path("/data/shared/exchange/uashehab/chembl35")


def fetch_activities():
    """Test fetching activity data from ChEMBL."""
    activities = fetch_activities_chembl(PATH_TO_CHEMBL)
    assert isinstance(activities, pd.DataFrame)
