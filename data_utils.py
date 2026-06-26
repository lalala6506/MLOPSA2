"""Data utility functions.

Contracts assumed by the test suite (adjust to match your own implementation):

load_csv(filepath):
    - raises FileNotFoundError if the path does not exist
    - raises ValueError if the file is empty (0 bytes OR header-only / no rows)
    - returns a pandas.DataFrame on success

clean_phone(phone):
    - strips all non-digit characters
    - drops a leading US country code "1" if the result is 11 digits
    - returns a consistent "XXX-XXX-XXXX" string
    - raises ValueError for anything that isn't exactly 10 usable digits
      (None, empty, too short, too long, no digits)

validate_email(email):
    - returns True for a syntactically valid email, False otherwise
    - non-strings and whitespace-padded values are False (no silent stripping)
"""

import os
import re

import pandas as pd

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def load_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File is empty: {filepath}") from exc
    if df.empty:
        raise ValueError(f"File contains no data rows: {filepath}")
    return df


def clean_phone(phone):
    if phone is None:
        raise ValueError("Phone number cannot be None")
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        raise ValueError(f"Invalid phone number: {phone!r}")
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def validate_email(email):
    if not isinstance(email, str):
        return False
    return bool(_EMAIL_RE.match(email))
