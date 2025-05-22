import sqlite3
from pathlib import Path
import pandas as pd

CHEMBL_ACTIVITY_QUERY = """
WITH valid_data AS (
    SELECT molregno, assay_id, standard_type, standard_relation,
           standard_value, standard_units, pchembl_value, data_validity_comment
    FROM activities
    WHERE standard_type IN ('Kd', 'Potency', 'AC50', 'IC50', 'Ki', 'EC50')
      AND standard_relation = '='
      AND standard_units   = 'nM'
)
SELECT
    valid_data.molregno,
    valid_data.assay_id,
    assays.chembl_id                     AS assay_chembl_id,
    assays.assay_type,
    assays.confidence_score,
    assays.relationship_type,
    assays.description                   AS assay_description,
    assays.bao_format,
    assays.tid,
    target_dictionary.chembl_id          AS target_chembl_id,
    target_dictionary.pref_name          AS target_name,
    valid_data.standard_type,
    valid_data.standard_relation,
    valid_data.standard_value,
    valid_data.standard_units,
    valid_data.pchembl_value,
    valid_data.data_validity_comment

FROM valid_data
LEFT JOIN assays USING (assay_id)
LEFT JOIN target_dictionary USING (tid);"""


CHEMBL_METADATA_QUERY = """
SELECT DISTINCT
    activities.activity_id,               -- activity ID (record_id) for exact matching
    activities.molregno,
    activities.assay_id,
    assays.tid,                           -- Target ID for additional precision
    molecule_dictionary.chembl_id         AS molecule_id,
    compound_structures.canonical_smiles  AS canonical_smiles,
    compound_structures.molfile           AS molecule,
    compound_records.compound_key         AS compound_key,
    docs.doc_id,
    docs.doi,
    docs.journal                          AS journal_name,
    docs.pubmed_id,
    docs.title                            AS publication_title,
    docs.authors,
    docs.year                             AS publication_year,
    docs.doc_type                         AS document_type,
    docs.src_id

FROM activities
LEFT JOIN molecule_dictionary     USING (molregno)
LEFT JOIN compound_structures     USING (molregno)
LEFT JOIN compound_records        USING (molregno)
LEFT JOIN assays                  USING (assay_id)
LEFT JOIN docs ON assays.doc_id = docs.doc_id

WHERE activities.standard_type IN ('Kd', 'Potency', 'AC50', 'IC50', 'Ki', 'EC50')
  AND activities.standard_relation = '='
  AND activities.standard_units    = 'nM';"""


def fetch_data_chembl(query: str, path_to_chembl: Path) -> pd.DataFrame:
    """Fetch query from ChEMBL."""
    con = sqlite3.connect(path_to_chembl)
    return pd.read_sql(query, con=con)


def fetch_activities_chembl(path_to_chembl: Path) -> pd.DataFrame:
    """Fetch activity data from ChEMBL."""
    return fetch_data_chembl(query=CHEMBL_ACTIVITY_QUERY, path_to_chembl=path_to_chembl)


def fetch_metadata_chembl(path_to_chembl: Path) -> pd.DataFrame:
    """Fetch metadata from ChEMBL."""
    return fetch_data_chembl(query=CHEMBL_METADATA_QUERY, path_to_chembl=path_to_chembl)
