"""Unit tests for data_utils.

Run with:  pytest test_data_utils.py -v
"""

import pandas as pd
import pytest

from data_utils import load_csv, clean_phone, validate_email


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------
class TestLoadCsv:
    def test_successful_loading(self, tmp_path):
        csv_file = tmp_path / "good.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        df = load_csv(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["name", "age"]
        assert df.loc[0, "name"] == "Alice"

    def test_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_csv(str(missing))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")  # zero bytes
        with pytest.raises(ValueError):
            load_csv(str(empty))

    def test_header_only_file(self, tmp_path):
        # A file with a header but no data rows is also "empty" of data.
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("name,age\n")
        with pytest.raises(ValueError):
            load_csv(str(header_only))


# ---------------------------------------------------------------------------
# clean_phone
# ---------------------------------------------------------------------------
class TestCleanPhone:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1234567890", "123-456-7890"),
            ("123-456-7890", "123-456-7890"),
            ("(123) 456-7890", "123-456-7890"),
            ("123.456.7890", "123-456-7890"),
            ("123 456 7890", "123-456-7890"),
            ("+1 (123) 456-7890", "123-456-7890"),   # leading country code
            ("1-123-456-7890", "123-456-7890"),
        ],
    )
    def test_valid_formats_normalized(self, raw, expected):
        assert clean_phone(raw) == expected

    def test_consistent_output_across_formats(self):
        # Different inputs representing the same number must collapse to one value.
        variants = ["1234567890", "(123) 456-7890", "123.456.7890", "+1 123 456 7890"]
        outputs = {clean_phone(v) for v in variants}
        assert outputs == {"123-456-7890"}

    @pytest.mark.parametrize(
        "bad",
        [
            "",                # empty string
            "12345",           # too short
            "123456789",       # 9 digits
            "123456789012",    # too long, not country-code form
            "abcdefghij",      # letters only -> 0 digits
            "phone number",    # no digits
            "555-CALL-NOW",    # mixed, not 10 digits
        ],
    )
    def test_invalid_inputs_raise(self, bad):
        with pytest.raises(ValueError):
            clean_phone(bad)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            clean_phone(None)


# ---------------------------------------------------------------------------
# validate_email
# ---------------------------------------------------------------------------
class TestValidateEmail:
    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "first.last@example.co.uk",
            "user+tag@example.org",
            "user_name@example-domain.com",
            "u@e.io",
            "123@numbers.net",
        ],
    )
    def test_valid_emails(self, email):
        assert validate_email(email) is True

    @pytest.mark.parametrize(
        "email",
        [
            "plainaddress",          # no @
            "@no-local.com",         # missing local part
            "user@",                 # missing domain
            "user@domain",           # no TLD
            "user@domain.c",         # TLD too short (1 char)
            "user@@example.com",     # double @
            "user @example.com",     # space in address
            "user@exa mple.com",     # space in domain
        ],
    )
    def test_invalid_emails(self, email):
        assert validate_email(email) is False

    @pytest.mark.parametrize(
        "edge",
        [
            "",                      # empty string
            "   ",                   # whitespace only
            " user@example.com",     # leading space
            "user@example.com ",     # trailing space
            None,                    # not a string
            12345,                   # not a string
            ["user@example.com"],    # wrong type entirely
        ],
    )
    def test_edge_cases_return_false(self, edge):
        assert validate_email(edge) is False
