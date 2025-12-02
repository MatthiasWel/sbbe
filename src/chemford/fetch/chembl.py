import sqlite3
from pathlib import Path
import pandas as pd

CHEMBL_QUERY = """
WITH valid_data AS (
    SELECT DISTINCT molregno, assay_id, standard_type, standard_value,
                    data_validity_comment, bao_endpoint AS bao_id, src_id,
                    pchembl_value, record_id
    FROM activities
    WHERE standard_type IN ('Kd', 'Potency', 'AC50', 'IC50', 'Ki', 'EC50')
      AND standard_relation = '='
      AND standard_units   = 'nM'
)
SELECT
    valid_data.molregno                 AS molecule_id,
    valid_data.src_id,
    valid_data.pchembl_value,
    valid_data.standard_type,
    valid_data.standard_value,
    valid_data.data_validity_comment,
    valid_data.record_id,

    assays.assay_id                     AS assay_id,
    assays.chembl_id                    AS assay_chembl_id,
    assays.description                  AS assay_description,
    assays.confidence_score,
    assays.relationship_type,
    assays.assay_type,
    assays.bao_format,
    assays.assay_strain                 AS assay_strain,
    assays.variant_id                   AS variant_id,
    assays.assay_category,
    assays.assay_tax_id,
    assays.assay_tissue,
    assays.assay_cell_type,
    assays.assay_subcellular_fraction,
    assays.src_assay_id,
    assays.curated_by,
    assays.aidx,
    assays.assay_group,

    curation_lookup.description         AS curation_description,

    target_dictionary.tid               AS target_id,
    target_dictionary.chembl_id         AS target_chembl_id,
    target_dictionary.pref_name         AS target_name,
    target_dictionary.organism          AS organism,
    target_dictionary.target_type       AS target_type,

    variant_sequences.variant_id,
    variant_sequences.accession         AS uniprot_accesion,

    source.src_description,
    source.src_comment,
    source.src_short_name,

    bioassay_ontology.bao_id            AS bao_id,
    bioassay_ontology.label             AS bao_label,

    docs.doc_id,
    docs.journal,
    docs.year,
    docs.volume,
    docs.issue,
    docs.doi,
    docs.title,
    docs.doc_type,
    docs.authors,

    compound_records.compound_key

FROM valid_data
LEFT JOIN assays                        USING (assay_id)
LEFT JOIN curation_lookup               USING (curated_by)
LEFT JOIN target_dictionary             USING (tid)
LEFT JOIN variant_sequences             USING (variant_id)
LEFT JOIN source                        USING (src_id)
LEFT JOIN bioassay_ontology             USING (bao_id)
LEFT JOIN docs                          USING (doc_id)
LEFT JOIN compound_records              USING (record_id)
;"""


def fetch_query_chembl(query: str, path_to_chembl: Path) -> pd.DataFrame:
    """Fetch query from ChEMBL."""
    con = sqlite3.connect(path_to_chembl)
    return pd.read_sql(query, con=con)


def fetch_data_chembl(path_to_chembl: Path) -> pd.DataFrame:
    """Fetch activity data from ChEMBL."""
    return fetch_query_chembl(query=CHEMBL_QUERY, path_to_chembl=path_to_chembl)
