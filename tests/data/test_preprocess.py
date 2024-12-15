import pytest
import yaml
from src.data.preprocess import clean_text, label_sentiment

with open("tests/data/fixtures/fixtures_preprocess.yaml", "r") as f:
    cases = yaml.safe_load(f)


def test_clean_text():
    """
    Test the clean_text function with various input scenarios.

    Args:
        clean_text_cases: Dictionary containing test cases loaded from YAML.
    """
    # Iterate over the test cases
    for case_name, case in cases["test_clean_text"].items():
        input_text = case["input"]
        expected_output = case["expected"]

        result = clean_text(input_text)

        # Assertions
        # 1. Check that the result is equal to the expected output
        assert result == expected_output, (
            f"{case_name}: Expected '{expected_output}', got '{result}' "
            f"for input '{input_text}'"
        )


def test_label_sentiment():
    """
    Test the label_sentiment function with various rating inputs.

    Args:
        label_sentiment_cases: Dictionary containing test cases loaded from YAML.
    """

    for case_name, case in cases["test_label_sentiment"].items():
        rating = case.get("rating")
        expected = case.get("expected")
        expect_exception = case.get("expect_exception", False)

        # Execute the label_sentiment function
        if expect_exception:
            with pytest.raises(Exception):
                label_sentiment(rating)
        else:
            result = label_sentiment(rating)

            # Assertions
            # 1. Check that the result is equal to the expected output
            assert (
                result == expected
            ), f"{case_name}: For rating '{rating}', expected '{expected}', got '{result}'"
