import yaml
from src.model.inference import SentimentInference


with open("tests/model/fixtures/fixtures_inference.yaml", "r") as f:
    cases = yaml.safe_load(f)

inference_engine = SentimentInference()


def test_sentiment_inference():
    """
    Test the split_data function with various scenarios.
    """
    # Iterate over the test cases
    for case_name, case in cases["test_sentiment_inference"].items():
        input_text = case["input_text"]
        expected = case["expected_output"]

        # Execute the predict_sentiment function
        predictions = inference_engine.predict_sentiment(input_text)

        # Get the predicted sentiment
        result = max(predictions, key=predictions.get)

        # Assertions
        # 1. Check that the result is equal to the expected output
        assert result == expected, f"{case_name}: Expected {expected}, got {result}"

        # 2. Check the predictions are a dictionary
        assert isinstance(
            predictions, dict
        ), f"{case_name}: Predictions are not a dictionary."
