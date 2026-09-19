# src

Reusable code — anything imported by more than one notebook belongs here rather
than being copy-pasted between them.

Suggested modules as the project develops:

| Module | Responsibility |
| --- | --- |
| `load.py` | Read the seven raw CSVs, normalize column names, concatenate with a day label |
| `clean.py` | Drop degenerate columns, handle infinities and NaNs, coerce dtypes |
| `features.py` | Feature selection — including the exclusion of host identifiers and timestamps |
| `sample.py` | Generate the committed `data/sample/` excerpt reproducibly |

## Conventions

- Read from `data/raw/`, write to `data/interim/` or `data/processed/`. Never
  modify anything in `data/raw/`.
- Take paths as arguments; do not hard-code absolute paths from anyone's machine.
- Any randomness takes an explicit seed.

## Column name warning

The raw CICIDS2017 CSVs have inconsistent whitespace in their headers — several
column names carry a leading space (`' Flow Duration'`). Strip whitespace from
headers on load, in one place, or every downstream `KeyError` will be a mystery.
