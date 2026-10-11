# Databricks notebook source
"""Generate synthetic company address records for the address matching pipeline.

Every generated company (entity) is stored in one to three source systems. The first
record of an entity is written in its canonical form. Additional records of the same
entity contain variations that occur in real source systems:

* typos (insertion, deletion, substitution, transposition of adjacent characters),
* spelling variants of the legal form (e.g. "GmbH", "Gmbh", "G.m.b.H."),
* spelling variants of the street type (e.g. "straße", "strasse", "str."),
* transliterated umlauts (e.g. "Müller" and "Mueller"),
* omitted legal forms.

Distinct entities share zip codes, and some share an address (office buildings,
business parks). This produces candidate pairs that must not be matched.

The column ``entity_id`` contains the ground truth. The matching pipeline does not use
it for matching; it is used only to measure precision and recall.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

NUM_ROWS = 500
SEED = 42

SYSTEMS = ["CRM", "ERP", "WEB", "LEGACY"]

# Number of records per entity and its probability weight.
RECORDS_PER_ENTITY = [1, 2, 3]
RECORDS_PER_ENTITY_WEIGHTS = [55, 30, 15]

# Probabilities of the variations applied to additional records of an entity.
NAME_TYPO_RATE = 0.4
STREET_TYPO_RATE = 0.2
UMLAUT_TRANSLITERATION_RATE = 0.3
LEGAL_FORM_OMISSION_RATE = 0.1

# Probability that a new entity is located at the address of an existing entity.
SHARED_ADDRESS_RATE = 0.1
# Probability that a new entity is a look-alike of an existing entity: same industry,
# same address, and a similar surname (e.g. "Meyer Logistik" and "Meier Logistik").
# Look-alikes are different companies and test the matching for false positives.
LOOKALIKE_RATE = 0.05

SIMILAR_SURNAMES = {
    "Meyer": "Meier",
    "Meier": "Meyer",
    "Hoffmann": "Hofmann",
    "Hofmann": "Hoffmann",
    "Schmidt": "Schmitt",
    "Schmitt": "Schmidt",
}
SURNAMES = [
    "Müller",
    "Schmidt",
    "Schmitt",
    "Schneider",
    "Fischer",
    "Weber",
    "Meyer",
    "Meier",
    "Wagner",
    "Becker",
    "Schulz",
    "Hoffmann",
    "Hofmann",
    "Koch",
    "Richter",
    "Klein",
    "Wolf",
    "Schröder",
    "Neumann",
    "Braun",
]
INDUSTRIES = [
    "Bau",
    "Metallbau",
    "Logistik",
    "Elektrotechnik",
    "Software",
    "Immobilien",
    "Handel",
    "Consulting",
    "Druck",
    "Maschinenbau",
    "Haustechnik",
    "Transporte",
]

# Each legal form maps to its spelling variants. The first variant is the canonical one.
LEGAL_FORMS = {
    "GmbH": ["GmbH", "Gmbh", "GMBH", "G.m.b.H."],
    "AG": ["AG", "Ag", "Aktiengesellschaft"],
    "KG": ["KG", "Kg"],
    "GmbH & Co. KG": ["GmbH & Co. KG", "GmbH & Co KG", "GmbH + Co. KG"],
    "e.K.": ["e.K.", "eK"],
}

STREETS = [
    "Haupt",
    "Bahnhof",
    "Garten",
    "Schul",
    "Kirch",
    "Berg",
    "Wald",
    "Linden",
    "Goethe",
    "Schiller",
    "Industrie",
    "Mühlen",
    "Friedrich",
    "Rosen",
    "Birken",
]

# Each street type maps to its spelling variants. The first variant is the canonical one.
STREET_TYPES = {
    "straße": ["straße", "strasse", "str.", " Str."],
    "weg": ["weg"],
    "allee": ["allee"],
    "gasse": ["gasse"],
    "ring": ["ring"],
    "platz": ["platz"],
}
STREET_TYPE_WEIGHTS = [50, 15, 10, 10, 10, 5]

ZIP_CODES = [
    "01067",
    "04109",
    "10115",
    "20095",
    "28195",
    "30159",
    "40213",
    "50667",
    "60311",
    "70173",
    "80331",
    "90402",
]

UMLAUTS = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}
)
LETTERS = "abcdefghijklmnopqrstuvwxyz"
COLUMNS = ["system_id", "record_id", "company_name", "street_name", "zip_code", "entity_id"]


@dataclass(frozen=True)
class Address:
    """Canonical address of a company."""

    street: str
    street_type: str
    house_number: str
    zip_code: str


@dataclass(frozen=True)
class Company:
    """Canonical attributes of a company (entity)."""

    surname: str
    industry: str
    legal_form: str
    address: Address

    @property
    def name(self) -> str:
        """Return the company name without the legal form."""
        return f"{self.surname} {self.industry}"


def add_typo(text: str, rng: random.Random) -> str:
    """Apply one random typo to a string.

    Args:
        text: The original string.
        rng: The random number generator.

    Returns:
        The string with one insertion, deletion, substitution, or transposition of two
        adjacent characters. Strings shorter than two characters are returned unchanged.
    """
    if len(text) < 2:
        return text
    chars = list(text)
    error_type = rng.choice(["insert", "delete", "substitute", "transpose"])
    # A transposition needs a following character.
    index = rng.randrange(len(chars) - 1 if error_type == "transpose" else len(chars))
    if error_type == "insert":
        chars.insert(index, rng.choice(LETTERS))
    elif error_type == "delete":
        del chars[index]
    elif error_type == "substitute":
        chars[index] = rng.choice(LETTERS.replace(chars[index].lower(), ""))
    else:
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


def create_address(rng: random.Random) -> Address:
    """Create a random address.

    Args:
        rng: The random number generator.

    Returns:
        A new address. About 10 % of the house numbers have a letter suffix ("12a").
    """
    house_number = str(rng.randint(1, 120))
    if rng.random() < 0.1:
        house_number += rng.choice("ab")
    return Address(
        street=rng.choice(STREETS),
        street_type=rng.choices(list(STREET_TYPES), weights=STREET_TYPE_WEIGHTS)[0],
        house_number=house_number,
        zip_code=rng.choice(ZIP_CODES),
    )


def create_company(rng: random.Random, existing: list[Company]) -> Company:
    """Create a company that differs from all existing companies.

    Args:
        rng: The random number generator.
        existing: The companies created so far.

    Returns:
        A new company. With the probability ``LOOKALIKE_RATE``, it is a look-alike of an
        existing company. With the probability ``SHARED_ADDRESS_RATE``, it is located at
        the address of an existing company. Otherwise, it has a new address.
    """
    lookalike_models = [c for c in existing if c.surname in SIMILAR_SURNAMES]
    while True:
        surname, industry = rng.choice(SURNAMES), rng.choice(INDUSTRIES)
        legal_form = rng.choice(list(LEGAL_FORMS))
        draw = rng.random()
        if lookalike_models and draw < LOOKALIKE_RATE:
            model = rng.choice(lookalike_models)
            surname, industry, address = (
                SIMILAR_SURNAMES[model.surname],
                model.industry,
                model.address,
            )
        elif existing and draw < LOOKALIKE_RATE + SHARED_ADDRESS_RATE:
            address = rng.choice(existing).address
        else:
            address = create_address(rng)
        company = Company(surname, industry, legal_form, address)
        # Two entities with the same name at the same address cannot be told apart.
        if not any(c.name == company.name and c.address == company.address for c in existing):
            return company


def render_record(company: Company, rng: random.Random, canonical: bool) -> tuple[str, str]:
    """Render the company name and the street name of one source record.

    Args:
        company: The company to render.
        rng: The random number generator.
        canonical: If True, the canonical spelling is used. Otherwise, variations
            are applied.

    Returns:
        A tuple of company name and street name (including the house number).
    """
    address = company.address
    name = company.name
    street = address.street
    legal_form = LEGAL_FORMS[company.legal_form][0]
    street_type = STREET_TYPES[address.street_type][0]

    if not canonical:
        legal_form = rng.choice(LEGAL_FORMS[company.legal_form])
        street_type = rng.choice(STREET_TYPES[address.street_type])
        if rng.random() < UMLAUT_TRANSLITERATION_RATE:
            name = name.translate(UMLAUTS)
            street = street.translate(UMLAUTS)
            street_type = street_type.translate(UMLAUTS)
        if rng.random() < NAME_TYPO_RATE:
            name = add_typo(name, rng)
        if rng.random() < STREET_TYPO_RATE:
            street = add_typo(street, rng)
        if rng.random() < LEGAL_FORM_OMISSION_RATE:
            legal_form = ""

    company_name = f"{name} {legal_form}".strip()
    street_name = f"{street}{street_type} {address.house_number}"
    return company_name, street_name


def generate_test_data(num_rows: int = NUM_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic address records with known duplicates.

    Args:
        num_rows: The number of records to generate.
        seed: The seed of the random number generator. The same seed produces the
            same data.

    Returns:
        A DataFrame with the columns ``system_id``, ``record_id``, ``company_name``,
        ``street_name``, ``zip_code``, and ``entity_id`` (ground truth).
    """
    rng = random.Random(seed)
    companies: list[Company] = []
    records: list[dict[str, str]] = []

    while len(records) < num_rows:
        company = create_company(rng, companies)
        companies.append(company)
        entity_id = f"E{len(companies):04d}"
        num_records = rng.choices(RECORDS_PER_ENTITY, weights=RECORDS_PER_ENTITY_WEIGHTS)[0]
        # Each record of an entity comes from a different source system.
        for position, system_id in enumerate(rng.sample(SYSTEMS, num_records)):
            company_name, street_name = render_record(company, rng, canonical=position == 0)
            records.append(
                {
                    "system_id": system_id,
                    "company_name": company_name,
                    "street_name": street_name,
                    "zip_code": company.address.zip_code,
                    "entity_id": entity_id,
                }
            )

    records = records[:num_rows]
    # Shuffle so that duplicates are not stored in adjacent rows, then assign record IDs.
    rng.shuffle(records)
    for number, record in enumerate(records, start=1):
        record["record_id"] = f"R{number:04d}"
    return pd.DataFrame(records, columns=COLUMNS)


def resolve_output_path() -> Path:
    """Return the path of the generated CSV file.

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


def main() -> None:
    """Generate the test data and write it to ``data/test_addresses.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    df = generate_test_data()
    output_path = resolve_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, lineterminator="\n")
    logger.info(
        "Generated %d records of %d entities at %s",
        len(df),
        df["entity_id"].nunique(),
        output_path,
    )


if __name__ == "__main__":
    main()
