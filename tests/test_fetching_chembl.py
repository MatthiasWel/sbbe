from pathlib import Path
import pandas as pd
from chemford.fetch.chembl import fetch_data_chembl

PATH_TO_CHEMBL = Path("/data/shared/exchange/uashehab/chembl35/chembl_35.db")


def test_fetch():
    """Test fetching data from ChEMBL."""
    data = fetch_data_chembl(PATH_TO_CHEMBL)
    assert isinstance(data, pd.DataFrame), "Fetched data is not a df"
    n_duplicates = len(data[data.duplicated(keep=False)])
    assert n_duplicates == 0, f"There are {n_duplicates} duplicates"
