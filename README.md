[![Deploy Databricks Asset Bundle](https://github.com/kleinnconrad/analytics.demo.levenshtein/actions/workflows/deploy_bundle.yml/badge.svg)](https://github.com/kleinnconrad/analytics.demo.levenshtein/actions/workflows/deploy_bundle.yml)

# Address Matching with the Levenshtein Distance

This repository is a demo and training project. It shows how the Levenshtein distance is used in master data management (MDM) to find duplicate company records across source systems, and how the result is evaluated. The pipeline runs locally and as a Databricks job.

## Table of Contents
* [Repository Structure](#repository-structure)
* [Getting Started](#getting-started)
* [Levenshtein Distance](#levenshtein-distance)
  * [Definition](#definition)
  * [Normalized Similarity](#normalized-similarity)
  * [Difference to Levenshtein.ratio](#difference-to-levenshteinratio)
* [Matching Pipeline](#matching-pipeline)
* [Test Data](#test-data)
* [Evaluation and Threshold Selection](#evaluation-and-threshold-selection)
* [Limitations in Master Data Management](#limitations-in-master-data-management)
* [Databricks Asset Bundles](#databricks-asset-bundles)
* [CI/CD Pipeline (GitHub Actions)](#cicd-pipeline-github-actions)

---

## Repository Structure

| Path | Content |
| --- | --- |
| `src/levenshtein.py` | Matching pipeline (local script and Databricks notebook) |
| `data_generation/generate_test_data.py` | Generator for the synthetic test data (local script and Databricks notebook) |
| `data/test_addresses.csv` | Generated test data (500 records) |
| `tests/` | Unit tests (pytest) |
| `databricks.yml`, `resources/` | Databricks Asset Bundle and job definition |
| `.github/` | GitHub Actions workflows and Dependabot configuration |

---

## Getting Started

Requirements: Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
# Install the dependencies into a local virtual environment
uv sync

# Optional: regenerate the test data (data/test_addresses.csv is part of the repository)
uv run python data_generation/generate_test_data.py

# Run the matching pipeline
uv run python src/levenshtein.py

# Run the tests and the code checks
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

New dependencies are added with `uv add <package>` (or `uv add --dev <package>` for development tools). This updates `pyproject.toml` and `uv.lock`. If the version of `Levenshtein` changes, the `%pip install` command in the first cell of `src/levenshtein.py` must be changed to the same version; a unit test checks this.

---

## Levenshtein Distance

### Definition
The Levenshtein distance between two strings is the minimum number of single-character edits that change one string into the other. The allowed edits are:

* **Insertion** (e.g. `cat` to `cart`)
* **Deletion** (e.g. `cart` to `cat`)
* **Substitution** (e.g. `cat` to `bat`)

Each edit costs 1. Example: `kitten` to `sitting` needs three edits (substitute `k` with `s`, substitute `e` with `i`, insert `g`), so the distance is 3.

### Normalized Similarity
The distance depends on the length of the strings: one edit in a 5-character name is a larger difference than one edit in a 30-character name. The pipeline therefore uses the normalized similarity:

```
similarity = 1 - distance / max(len(a), len(b))
```

The value is 1.0 for identical strings. Each edit lowers it by `1 / max(len(a), len(b))`. Examples:

| a | b | Distance | Similarity |
| --- | --- | --- | --- |
| `kitten` | `sitting` | 3 | 0.571 |
| `meyer` | `meier` | 1 | 0.800 |
| `muellerbau` | `mullerbau` | 1 | 0.900 |

If one of the strings is empty, the pipeline returns 0.0, so that two missing values do not count as a match.

### Difference to Levenshtein.ratio
The `Levenshtein` library also provides `Levenshtein.ratio()`. This function does not use the Levenshtein distance. It uses the Indel distance, which allows only insertions and deletions; a substitution therefore costs 2. The result is normalized with the sum of both lengths: `1 - indel_distance / (len(a) + len(b))`. For `kitten` and `sitting`, `ratio()` returns 0.615 instead of 0.571. The pipeline uses `Levenshtein.distance()` so that the computed value matches the definition above.

---

## Matching Pipeline

`src/levenshtein.py` processes the records in the following steps:

1. **Loading:** Reads `data/test_addresses.csv`. All columns are read as strings, so that zip codes keep leading zeros (`01067`). The path is resolved relative to the script locally and relative to the notebook folder in Databricks.
2. **Standardization:**
   * Company names: lowercase, transliteration of umlauts (`ü` to `ue`, `ß` to `ss`), removal of legal forms as whole words (`GmbH`, `G.m.b.H.`, `AG`, `GmbH & Co. KG`, `e.K.`, ...), removal of all characters other than letters and digits. Example: `Müller Bau GmbH & Co. KG` becomes `muellerbau`.
   * Street names: same normalization, removal of the house number, abbreviation of `strasse` to `str`. Other street types (`weg`, `gasse`, `ring`, ...) are kept because `Lindenweg` and `Lindengasse` are different streets. Example: `Haupt Straße 12a` becomes `hauptstr`.
   * House numbers: extracted separately (`12a`).
3. **Blocking:** Only records with the same zip code are compared. Each pair is created once (A with B, not also B with A). For the test data, this reduces the number of comparisons from 124,750 to about 11,000.
4. **Scoring:** Calculates the normalized Levenshtein similarity of the standardized company names and street names.
5. **Classification:** A pair is a duplicate if the name similarity and the street similarity are both at least 0.80 and the house numbers are identical. House numbers are compared exactly, because `12` and `13` differ by only one edit but are different addresses.
6. **Evaluation:** If the input contains the ground truth column `entity_id`, the pipeline logs precision, recall, and F1 score, and a table for a range of thresholds (see [Evaluation and Threshold Selection](#evaluation-and-threshold-selection)).
7. **Output:** In Databricks, the duplicate pairs are written to the Delta table `workspace.default.address_duplicates`. The table contains the system ID, record ID, company name, and street name of both records, the zip code, and both similarity scores. Locally, the result is only logged.

---

## Test Data

`data_generation/generate_test_data.py` generates 500 records of German companies. The random number generator uses a fixed seed, so every run produces the same file.

* **Entities:** Each company (entity) is stored in one to three source systems (`CRM`, `ERP`, `WEB`, `LEGACY`).
* **Variations:** The first record of an entity uses the canonical spelling. Further records contain typos (insertion, deletion, substitution, or transposition of two adjacent characters), spelling variants of the legal form (`GmbH`, `Gmbh`, `G.m.b.H.`) and of the street type (`straße`, `strasse`, `str.`), transliterated umlauts (`Müller`, `Mueller`), and omitted legal forms.
* **Difficult non-duplicates:** All records are distributed over 12 zip codes, so each block contains many different companies. Some companies share an address with another company. Some of these are look-alikes with a similar surname and the same industry (`Meyer Logistik` and `Meier Logistik`); they are different companies and must not be matched.
* **Ground truth:** The column `entity_id` identifies the entity of each record. The matching does not use this column; it is used only for the evaluation.

---

## Evaluation and Threshold Selection

The evaluation compares the found pairs with the true pairs from `entity_id`:

* **Precision:** share of found pairs that are true duplicates. Low precision means records of different companies would be merged.
* **Recall:** share of true duplicate pairs that were found. Low recall means duplicates remain in the data.
* **F1 score:** harmonic mean of precision and recall.

Output for the test data (the same threshold is used for names and streets):

```
 threshold  pairs  precision  recall    f1
      0.70    266      0.868   0.996 0.928
      0.75    258      0.891   0.991 0.939
      0.80    253      0.889   0.970 0.928
      0.85    237      0.886   0.905 0.896
      0.90    195      0.882   0.741 0.806
      0.95    105      0.981   0.444 0.611
      1.00     99      1.000   0.427 0.598
```

* A higher threshold increases precision and lowers recall. The threshold 0.80 finds 97 % of the true pairs with a precision of 0.889.
* At the threshold 0.80, all false positives are look-alike companies at the same address (`Meyer`/`Meier`, `Hoffmann`/`Hofmann`, `Schmidt`/`Schmitt`). Their name similarity is between 0.85 and 0.95. A threshold of 0.95 excludes most of them but also removes more than half of the true pairs.
* Below 0.75, companies with different surnames and a long common word are also matched: `Schneider Maschinenbau` and `Wagner Maschinenbau` have a similarity of 0.714, because the common part `maschinenbau` makes up most of the string.
* Most false negatives contain a transposition in a short street name (`Waldgasse`/`Wladgasse`, similarity 0.778).
* The values apply to the synthetic data only. For real data, the thresholds must be determined with a manually labeled sample.

---

## Limitations in Master Data Management

* **Transpositions:** Swapping two adjacent characters counts as two edits. The Damerau-Levenshtein distance counts it as one.
* **Short strings:** One edit in a 5-character string lowers the similarity by 0.2. Short names need lower thresholds or a different measure (e.g. Jaro-Winkler).
* **Similar names of different companies:** `Meyer Logistik` and `Meier Logistik` differ by one character. String similarity cannot distinguish them; additional attributes (tax ID, commercial register number, phone number) or a manual review are required.
* **Word order and abbreviations:** `Bau Müller` and `Müller Bau`, or `Intl.` and `International`, have a large edit distance. Token-based measures and abbreviation dictionaries handle these cases.
* **Blocking errors:** Records with different or missing zip codes are never compared. A typo in the zip code therefore causes a missed duplicate. Several blocking keys (e.g. zip code, and city plus first letters of the name) reduce this risk.
* **Pairs instead of entities:** The pipeline returns pairs. If A matches B and B matches C, A and C belong to the same entity even if they were not matched directly. Building entities (clusters) and a golden record per entity are separate MDM steps.
* **Runtime:** The number of comparisons grows quadratically with the block size. Large blocks need a finer blocking key.

---

## Databricks Asset Bundles

The Databricks resources are defined with Databricks Asset Bundles (DABs), the infrastructure-as-code format of Databricks. `databricks.yml` defines the bundle and the workspace; `resources/address_matching_job.yml` defines the job **Address Matching Pipeline** with two notebook tasks:

1. `data_generation_task` runs `data_generation/generate_test_data.py` and writes `data/test_addresses.csv` into the deployed bundle folder.
2. `deduplication_task` runs `src/levenshtein.py` after the first task and writes the result table.

Both Python files start with the line `# Databricks notebook source`, which makes Databricks import them as notebooks. The job does not define a cluster, so it runs on serverless compute. The `Levenshtein` library is installed with `%pip install` in the first notebook cell.

Manual deployment with a configured Databricks CLI:

```bash
databricks bundle validate
databricks bundle deploy -t default
databricks bundle run address_matching_pipeline -t default
```

---

## CI/CD Pipeline (GitHub Actions)

The workflow `.github/workflows/deploy_bundle.yml` contains two jobs:

* **Run Tests and Linting:** Runs for every pull request, for pushes to `main` that change relevant files, and on manual start. It installs the locked dependencies (`uv sync --locked`, which fails if `uv.lock` does not match `pyproject.toml`), checks the code with `ruff check` and `ruff format --check`, and runs the unit tests with `pytest`. The tests run on the GitHub runner and do not use Databricks compute.
* **Validate and Deploy Bundle:** Runs only on `main` after the first job has passed. It runs `databricks bundle validate` and `databricks bundle deploy`. The CLI authenticates with the repository secret `DATABRICKS_TOKEN` (a Databricks personal access token).

Dependabot checks the Python dependencies (`uv` ecosystem, updates `pyproject.toml` and `uv.lock`) and the GitHub Actions weekly. The workflow `.github/workflows/approve-dependabot.yml` approves pull requests opened by Dependabot. Merging remains a manual step.
