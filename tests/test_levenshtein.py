"""Unit tests for the address matching pipeline."""

import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from data_generation.generate_test_data import generate_test_data
from src.levenshtein import (
    build_candidate_pairs,
    clean_company_name,
    clean_street_name,
    evaluate,
    extract_house_number,
    levenshtein_similarity,
    score_pairs,
    select_duplicates,
    standardize_records,
    true_duplicate_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_records(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Create standardized records from (record_id, company_name, street_name, zip_code)."""
    records = pd.DataFrame(rows, columns=["record_id", "company_name", "street_name", "zip_code"])
    records["system_id"] = "CRM"
    return standardize_records(records)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("TechCorp GmbH", "techcorp"),
        ("Big Business AG", "bigbusiness"),
        ("A.B.C. mbH", "abc"),
        ("My*Company", "mycompany"),
        ("Müller Bau GmbH & Co. KG", "muellerbau"),
        ("Mueller Bau G.m.b.H.", "muellerbau"),
        ("Schröder Druck e.K.", "schroederdruck"),
        ("Acme Inc.", "acme"),
        # Legal forms are removed only as whole words, not inside other words.
        ("Hagen Bau AG", "hagenbau"),
        ("Imagination GmbH", "imagination"),
        ("Kgalagadi Handel", "kgalagadihandel"),
    ],
)
def test_clean_company_name(name: str, expected: str) -> None:
    """Legal forms, punctuation, and whitespace are removed; umlauts are transliterated."""
    assert clean_company_name(name) == expected


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan")])
def test_clean_company_name_missing(missing: object) -> None:
    """Missing values become empty strings."""
    assert clean_company_name(missing) == ""


@pytest.mark.parametrize(
    ("street", "expected"),
    [
        ("Hauptstraße 12", "hauptstr"),
        ("Hauptstrasse 12", "hauptstr"),
        ("Hauptstr. 12", "hauptstr"),
        ("Haupt Str. 12a", "hauptstr"),
        ("Lindenweg 5", "lindenweg"),
        ("Am Ring 5", "amring"),
        ("Avenue 12345", "avenue"),
        # Street types are not removed: "Ringstr." must not become an empty string.
        ("Ringstr. 100", "ringstr"),
        ("Wegener Weg 3", "wegenerweg"),
    ],
)
def test_clean_street_name(street: str, expected: str) -> None:
    """Spelling variants of "straße" are unified; house numbers are removed."""
    assert clean_street_name(street) == expected


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan")])
def test_clean_street_name_missing(missing: object) -> None:
    """Missing values become empty strings."""
    assert clean_street_name(missing) == ""


@pytest.mark.parametrize(
    ("street", "expected"),
    [
        ("Hauptstraße 12", "12"),
        ("Hauptstr. 12a", "12a"),
        ("Hauptstr. 12 a", "12a"),
        ("Straße des 17. Juni 5", "5"),
        ("Hauptstraße", ""),
        (None, ""),
    ],
)
def test_extract_house_number(street: str | None, expected: str) -> None:
    """The last number including an optional letter suffix is the house number."""
    assert extract_house_number(street) == expected


def test_levenshtein_similarity() -> None:
    """The similarity is 1 - distance / length of the longer string."""
    assert levenshtein_similarity("kitten", "sitting") == pytest.approx(1 - 3 / 7)
    assert levenshtein_similarity("meyer", "meier") == pytest.approx(0.8)
    assert levenshtein_similarity("abc", "abc") == 1.0
    # Empty values never count as a match.
    assert levenshtein_similarity("", "") == 0.0
    assert levenshtein_similarity("abc", "") == 0.0


def test_build_candidate_pairs_blocks_on_zip_code() -> None:
    """Only records with the same zip code are paired, and every pair appears once."""
    records = make_records(
        [
            ("R1", "Müller Bau GmbH", "Hauptstraße 1", "01067"),
            ("R2", "Mueller Bau GmbH", "Hauptstr. 1", "01067"),
            ("R3", "Müller Bau GmbH", "Hauptstraße 1", "01067"),
            ("R4", "Müller Bau GmbH", "Hauptstraße 1", "80331"),
            ("R5", "Müller Bau GmbH", "Hauptstraße 1", None),
        ]
    )
    pairs = build_candidate_pairs(records)
    assert set(zip(pairs["record_id_x"], pairs["record_id_y"], strict=True)) == {
        ("R1", "R2"),
        ("R1", "R3"),
        ("R2", "R3"),
    }


def test_select_duplicates() -> None:
    """Pairs need similar names, similar streets, and the same house number."""
    records = make_records(
        [
            ("R1", "Müller Bau GmbH", "Hauptstraße 12", "01067"),
            ("R2", "Mueller Bau G.m.b.H.", "Haupt Str. 12", "01067"),
            ("R3", "Müller Bau GmbH", "Hauptstraße 13", "01067"),
            ("R4", "Schmidt Logistik AG", "Hauptstraße 12", "01067"),
        ]
    )
    duplicates = select_duplicates(score_pairs(build_candidate_pairs(records)))
    assert list(zip(duplicates["record_id_x"], duplicates["record_id_y"], strict=True)) == [
        ("R1", "R2")
    ]
    assert duplicates.loc[0, "name_similarity"] == 1.0


def test_select_duplicates_without_candidates() -> None:
    """An empty set of candidate pairs produces an empty result instead of an error."""
    records = make_records([("R1", "Müller Bau GmbH", "Hauptstraße 12", "01067")])
    duplicates = select_duplicates(score_pairs(build_candidate_pairs(records)))
    assert duplicates.empty


def test_evaluate() -> None:
    """Precision, recall, and F1 are calculated from found and true pairs."""
    duplicates = pd.DataFrame({"record_id_x": ["R1", "R1"], "record_id_y": ["R2", "R3"]})
    metrics = evaluate(duplicates, {("R1", "R2"), ("R4", "R5")})
    assert metrics == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_pipeline_quality_on_generated_data() -> None:
    """The pipeline finds most duplicates of the generated data with few false positives."""
    records = standardize_records(generate_test_data())
    duplicates = select_duplicates(score_pairs(build_candidate_pairs(records)))
    metrics = evaluate(duplicates, true_duplicate_pairs(records))
    assert metrics["precision"] >= 0.85
    assert metrics["recall"] >= 0.9


def test_notebook_pins_same_levenshtein_version_as_pyproject() -> None:
    """The %pip install command of the Databricks notebook matches pyproject.toml."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pinned = next(
        dep for dep in pyproject["project"]["dependencies"] if dep.startswith("Levenshtein")
    )
    notebook = (PROJECT_ROOT / "src" / "levenshtein.py").read_text(encoding="utf-8")
    match = re.search(r"^# MAGIC %pip install (\S+)$", notebook, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == pinned
