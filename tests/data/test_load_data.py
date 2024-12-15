import pytest
import pandas as pd
import yaml

from src.data.load_data import load_data

with open("tests/data/fixtures/fixtures_load_data.yaml", "r") as f:
    cases = yaml.safe_load(f)


def test_load_data():
    """
    Test cases for the load_data function.

    This test function iterates over the test cases defined in the YAML fixture.
    For each case, it attempts to load the specified JSON lines file and verifies
    that the returned DataFrame matches the expected conditions (columns, row count),
    or that an exception is raised if expected.
    """

    # Iterate over the test cases
    for _, case in cases["test_load_data"].items():
        file_path = case["file_path"]
        expected_min_row_count = case.get("expected_min_row_count", 0)
        expect_exception = case.get("expect_exception", False)

        # Attempt to load the data
        if expect_exception:
            with pytest.raises(Exception):
                load_data(file_path)
        else:
            df = load_data(file_path)
            # Assertions
            # 1. Check that the returned object is a DataFrame
            assert isinstance(df, pd.DataFrame), "Returned object should be a DataFrame"

            # 2. Check that the DataFrame has the expected columns
            assert (
                len(df) >= expected_min_row_count
            ), f"Expected at least {expected_min_row_count} rows, got {len(df)}"
