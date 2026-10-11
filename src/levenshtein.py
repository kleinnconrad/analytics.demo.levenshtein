# Databricks notebook source
# MAGIC %pip install Levenshtein==0.27.5

# COMMAND ----------

"""Find duplicate company records with the Levenshtein distance.

The pipeline runs locally and as a Databricks notebook:

1. Load the address records from ``data/test_addresses.csv``.
2. Standardize company names, street names, and house numbers.
3. Build candidate pairs of records with the same zip code (blocking).
4. Score each candidate pair with the normalized Levenshtein similarity.
5. Keep the pairs above the similarity thresholds that have the same house number.
6. Evaluate the result against the ground truth column ``entity_id``, if present.
7. Save the duplicate pairs to a Delta table (Databricks only).
"""

import logging
import re
import unicodedata
from pathlib import Path

import Levenshtein
import pandas as pd

logger = logging.getLogger(__name__)

NAME_THRESHOLD = 0.80
STREET_THRESHOLD = 0.80
OUTPUT_TABLE = "workspace.default.address_duplicates"

# Tokens of legal forms that do not identify a company, e.g. "GmbH & Co. KG".
LEGAL_FORM_TOKENS = frozenset(
    {
        "ag",
        "aktiengesellschaft",
        "co",
        "corp",
        "ek",
        "gbr",
        "gmbh",
        "haftungsbeschraenkt",
        "inc",
        "kg",
        "kgaa",
        "llc",
        "ltd",
        "mbh",
        "ohg",
        "se",
        "ug",
    }
)
UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
# A house number is a number with an optional single letter suffix, e.g. "12" or "12a".
HOUSE_NUMBER = re.compile(r"(\d+)\s*([a-z])?\b")

OUTPUT_COLUMNS = [
    "system_id_x",
    "record_id_x",
    "company_name_x",
    "street_name_x",
    "system_id_y",
    "record_id_y",
    "company_name_y",
    "street_name_y",
    "zip_code",
    "name_similarity",
    "street_similarity",
]
OUTPUT_SCHEMA = (
    "system_id_x string, record_id_x string, company_name_x string, street_name_x string, "
    "system_id_y string, record_id_y string, company_name_y string, street_name_y string, "
    "zip_code string, name_similarity double, street_similarity double"
)


def normalize_text(text: str) -> str:
    """Convert a string to lowercase ASCII.

    German umlauts are transliterated ("ü" becomes "ue", "ß" becomes "ss") so that both
    spellings of a name match. Diacritics of other characters are removed ("é" becomes "e").

    Args:
        text: The original string.

    Returns:
        The normalized string.
    """
    text = unicodedata.normalize("NFC", text.lower()).translate(UMLAUTS)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def clean_company_name(name: str | None) -> str:
    """Standardize a company name for comparison.

    The name is normalized, legal forms are removed as whole words, and all characters
    other than letters and digits are removed. Example: "Müller Bau GmbH & Co. KG"
    becomes "muellerbau".

    Args:
        name: The original company name.

    Returns:
        The standardized company name, or an empty string for missing values.
    """
    if pd.isna(name):
        return ""
    # Remove dots first so that abbreviations such as "G.m.b.H." form one token.
    text = normalize_text(str(name)).replace(".", "")
    tokens = re.split(r"[^a-z0-9]+", text)
    return "".join(token for token in tokens if token not in LEGAL_FORM_TOKENS)


def clean_street_name(street: str | None) -> str:
    """Standardize a street name for comparison.

    The street name is normalized, the house number is removed, "strasse" is abbreviated
    to "str", and all characters other than letters are removed. Other street types
    ("weg", "gasse", "ring", ...) are kept because they distinguish different streets.
    Example: "Haupt Straße 12a" becomes "hauptstr".

    Args:
        street: The original street name including the house number.

    Returns:
        The standardized street name, or an empty string for missing values.
    """
    if pd.isna(street):
        return ""
    text = HOUSE_NUMBER.sub(" ", normalize_text(str(street)))
    text = re.sub(r"strasse\b", "str", text)
    return re.sub(r"[^a-z]", "", text)


def extract_house_number(street: str | None) -> str:
    """Extract the house number from a street name.

    Args:
        street: The original street name including the house number.

    Returns:
        The last house number in the string without spaces (e.g. "12a" for
        "Hauptstr. 12 a"), or an empty string if there is none.
    """
    if pd.isna(street):
        return ""
    matches = HOUSE_NUMBER.findall(normalize_text(str(street)))
    return "".join(matches[-1]) if matches else ""


def levenshtein_similarity(a: str, b: str) -> float:
    """Calculate the normalized Levenshtein similarity of two strings.

    The similarity is ``1 - distance / max(len(a), len(b))``. A value of 1.0 means the
    strings are identical; each edit lowers the value by ``1 / max(len(a), len(b))``.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The similarity in the range [0.0, 1.0]. If one of the strings is empty, the
        result is 0.0, so that missing values never count as a match.
    """
    if not a or not b:
        return 0.0
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b))


def resolve_data_path() -> Path:
    """Return the path of the input CSV file.

    Returns:
        The path ``data/test_addresses.csv`` in the repository root.
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
    except NameError:
        # Databricks notebooks do not define __file__. The working directory of a
        # notebook is the folder that contains it.
        project_root = Path("..")
    return project_root / "data" / "test_addresses.csv"


def load_records(path: Path) -> pd.DataFrame:
    """Load the address records from a CSV file.

    All columns are read as strings. This keeps leading zeros of zip codes ("01067")
    and prevents IDs from being parsed as numbers.

    Args:
        path: The path of the CSV file.

    Returns:
        The address records.
    """
    return pd.read_csv(path, dtype=str)


def standardize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Add the standardized columns used for matching.

    Args:
        records: The address records.

    Returns:
        A copy of the records with the columns ``clean_name``, ``clean_street``, and
        ``house_number``.
    """
    records = records.copy()
    records["clean_name"] = records["company_name"].map(clean_company_name)
    records["clean_street"] = records["street_name"].map(clean_street_name)
    records["house_number"] = records["street_name"].map(extract_house_number)
    return records


def build_candidate_pairs(records: pd.DataFrame) -> pd.DataFrame:
    """Build all pairs of records with the same zip code (blocking).

    Comparing every record with every other record requires n * (n - 1) / 2
    comparisons. Blocking compares only records within the same zip code. Each pair is
    returned once (record A with record B, not also B with A), and records are not
    paired with themselves. Records without a zip code cannot be blocked and are
    skipped.

    Args:
        records: The standardized address records.

    Returns:
        The candidate pairs. Columns of the first record have the suffix ``_x``,
        columns of the second record have the suffix ``_y``.
    """
    blockable = records[records["zip_code"].notna()]
    pairs = blockable.merge(blockable, on="zip_code", suffixes=("_x", "_y"))
    return pairs[pairs["record_id_x"] < pairs["record_id_y"]].reset_index(drop=True)


def score_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add the similarity scores to the candidate pairs.

    Args:
        pairs: The candidate pairs.

    Returns:
        A copy of the pairs with the columns ``name_similarity``, ``street_similarity``,
        and ``same_house_number``.
    """
    pairs = pairs.copy()
    pairs["name_similarity"] = [
        levenshtein_similarity(a, b)
        for a, b in zip(pairs["clean_name_x"], pairs["clean_name_y"], strict=True)
    ]
    pairs["street_similarity"] = [
        levenshtein_similarity(a, b)
        for a, b in zip(pairs["clean_street_x"], pairs["clean_street_y"], strict=True)
    ]
    # House numbers are compared exactly: "12" and "13" differ by one edit but are
    # different addresses.
    pairs["same_house_number"] = pairs["house_number_x"] == pairs["house_number_y"]
    return pairs


def select_duplicates(
    scored_pairs: pd.DataFrame,
    name_threshold: float = NAME_THRESHOLD,
    street_threshold: float = STREET_THRESHOLD,
) -> pd.DataFrame:
    """Select the candidate pairs that are classified as duplicates.

    Args:
        scored_pairs: The candidate pairs with similarity scores.
        name_threshold: The minimum similarity of the company names.
        street_threshold: The minimum similarity of the street names.

    Returns:
        The duplicate pairs with the columns in ``OUTPUT_COLUMNS``.
    """
    is_duplicate = (
        (scored_pairs["name_similarity"] >= name_threshold)
        & (scored_pairs["street_similarity"] >= street_threshold)
        & scored_pairs["same_house_number"]
    )
    return scored_pairs.loc[is_duplicate, OUTPUT_COLUMNS].reset_index(drop=True)


def true_duplicate_pairs(records: pd.DataFrame) -> set[tuple[str, str]]:
    """Return all pairs of records that belong to the same entity (ground truth).

    Args:
        records: The address records with the column ``entity_id``.

    Returns:
        The pairs as tuples ``(record_id_x, record_id_y)`` with
        ``record_id_x < record_id_y``.
    """
    pairs = records.merge(records, on="entity_id", suffixes=("_x", "_y"))
    pairs = pairs[pairs["record_id_x"] < pairs["record_id_y"]]
    return set(zip(pairs["record_id_x"], pairs["record_id_y"], strict=True))


def evaluate(duplicates: pd.DataFrame, true_pairs: set[tuple[str, str]]) -> dict[str, float]:
    """Compare the found duplicate pairs with the ground truth.

    Args:
        duplicates: The duplicate pairs found by the pipeline.
        true_pairs: The ground truth pairs from ``true_duplicate_pairs``.

    Returns:
        The precision (share of found pairs that are correct), the recall (share of
        true pairs that were found), and the F1 score (harmonic mean of both).
    """
    found_pairs = set(zip(duplicates["record_id_x"], duplicates["record_id_y"], strict=True))
    true_positives = len(found_pairs & true_pairs)
    precision = true_positives / len(found_pairs) if found_pairs else 0.0
    recall = true_positives / len(true_pairs) if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def log_threshold_sweep(scored_pairs: pd.DataFrame, true_pairs: set[tuple[str, str]]) -> None:
    """Log precision, recall, and F1 score for a range of thresholds.

    The same threshold is used for company names and street names.

    Args:
        scored_pairs: The candidate pairs with similarity scores.
        true_pairs: The ground truth pairs.
    """
    rows = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
        duplicates = select_duplicates(scored_pairs, threshold, threshold)
        rows.append(
            {"threshold": threshold, "pairs": len(duplicates)} | evaluate(duplicates, true_pairs)
        )
    logger.info("Threshold sweep:\n%s", pd.DataFrame(rows).round(3).to_string(index=False))


def save_to_delta(duplicates: pd.DataFrame, table: str = OUTPUT_TABLE) -> None:
    """Save the duplicate pairs to a Delta table if a Spark session is available.

    Args:
        duplicates: The duplicate pairs.
        table: The fully qualified name of the target table.
    """
    # Databricks notebooks provide the global variable "spark". It does not exist locally.
    spark = globals().get("spark")
    if spark is None:
        logger.info("No Spark session found (local execution). Results were not saved.")
        return
    (
        spark.createDataFrame(duplicates, schema=OUTPUT_SCHEMA)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )
    logger.info("Saved %d duplicate pairs to the Delta table %s", len(duplicates), table)


def main() -> None:
    """Run the address matching pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    records = standardize_records(load_records(resolve_data_path()))
    num_records = len(records)
    logger.info("Loaded %d records", num_records)

    pairs = build_candidate_pairs(records)
    logger.info(
        "Blocking on zip_code: %d candidate pairs instead of %d for a comparison of all records",
        len(pairs),
        num_records * (num_records - 1) // 2,
    )

    scored_pairs = score_pairs(pairs)
    duplicates = select_duplicates(scored_pairs)
    logger.info(
        "Duplicate pairs found (name >= %.2f, street >= %.2f, same house number): %d",
        NAME_THRESHOLD,
        STREET_THRESHOLD,
        len(duplicates),
    )
    logger.info("Sample duplicate pairs:\n%s", duplicates.head(10).round(3).to_string(index=False))

    if "entity_id" in records.columns:
        true_pairs = true_duplicate_pairs(records)
        metrics = evaluate(duplicates, true_pairs)
        logger.info(
            "Evaluation against %d true pairs: precision %.3f, recall %.3f, F1 %.3f",
            len(true_pairs),
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        log_threshold_sweep(scored_pairs, true_pairs)

    save_to_delta(duplicates)


if __name__ == "__main__":
    main()
