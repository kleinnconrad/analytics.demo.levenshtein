"""Unit tests for the synthetic test data generator."""

import random

import pandas as pd

from data_generation.generate_test_data import COLUMNS, add_typo, generate_test_data


def test_generate_test_data_shape_and_keys() -> None:
    """The data has the requested size, the expected columns, and unique record IDs."""
    df = generate_test_data(num_rows=200)
    assert len(df) == 200
    assert list(df.columns) == COLUMNS
    assert df["record_id"].is_unique
    assert df["zip_code"].str.fullmatch(r"\d{5}").all()


def test_generate_test_data_contains_duplicates() -> None:
    """Some entities are stored in more than one source system."""
    df = generate_test_data(num_rows=200)
    records_per_entity = df.groupby("entity_id").size()
    assert (records_per_entity > 1).any()
    # Records of the same entity come from different source systems.
    assert not df.duplicated(subset=["entity_id", "system_id"]).any()


def test_generate_test_data_is_reproducible() -> None:
    """The same seed produces the same data; a different seed produces different data."""
    pd.testing.assert_frame_equal(generate_test_data(seed=1), generate_test_data(seed=1))
    assert not generate_test_data(seed=1).equals(generate_test_data(seed=2))


def test_add_typo_changes_length_by_at_most_one() -> None:
    """A typo is a single edit operation."""
    rng = random.Random(0)
    for _ in range(100):
        typo = add_typo("Schneider", rng)
        assert abs(len(typo) - len("Schneider")) <= 1
    assert add_typo("a", rng) == "a"
