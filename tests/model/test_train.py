import yaml
import pandas as pd
from src.model.train import split_data, create_vectors, encode_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

with open("tests/model/fixtures/fixtures_train.yaml", "r") as f:
    cases = yaml.safe_load(f)


def test_split_data():
    """
    Test the split_data function with various scenarios.
    """
    # Iterate over the test cases
    for case_name, case in cases["test_split_data"].items():
        df_data = case["df"]
        test_size = case.get("test_size", 0.3)
        random_state = case.get("random_state", 42)
        expected = case["expected"]

        # Create DataFrame from the YAML data
        df = pd.DataFrame(df_data)

        # Execute the split_data function
        X_train, X_test, y_train, y_test = split_data(
            df, test_size=test_size, random_state=random_state
        )

        # Assertions
        # 1. Check the size of the training set
        assert (
            len(X_train) == expected["train_size"]
        ), f"{case_name}: Expected {expected['train_size']} training samples, got {len(X_train)}"

        # 2. Check the size of the testing set
        assert (
            len(X_test) == expected["test_size"]
        ), f"{case_name}: Expected {expected['test_size']} testing samples, got {len(X_test)}"


def test_create_vectors():
    """
    Test the create_vectors function with predefined scenarios.
    """
    # Iterate over the test cases
    for case_name, case in cases["test_create_vectors"].items():
        X_train = pd.Series(case["X_train"])
        X_test = pd.Series(case["X_test"])
        max_features = case.get("max_features", 5000)
        expected = case["expected"]

        # Execute the create_vectors function
        vectorizer, X_train_vec, X_test_vec = create_vectors(
            X_train, X_test, max_features=max_features
        )

        # Assertions
        # 1. Check that vectorizer is an instance of TfidfVectorizer
        assert isinstance(
            vectorizer, TfidfVectorizer
        ), f"{case_name}: Vectorizer is not an instance of TfidfVectorizer."

        # 2. Check the shape of X_train_vec
        assert X_train_vec.shape == tuple(
            expected["train_shape"]
        ), f"{case_name}: Expected train_shape {expected['train_shape']}, got {X_train_vec.shape}"

        # 3. Check the shape of X_test_vec
        assert X_test_vec.shape == tuple(
            expected["test_shape"]
        ), f"{case_name}: Expected test_shape {expected['test_shape']}, got {X_test_vec.shape}"

        # 4. Check the number of features
        assert (
            X_train_vec.shape[1] <= max_features
        ), f"{case_name}: Number of features {X_train_vec.shape[1]} exceeds max_features {max_features}"


def test_encode_labels():
    """
    Test the encode_labels function with predefined scenarios.
    """
    # Iterate over the test cases
    for case_name, case in cases["test_encode_labels"].items():
        y_train = pd.Series(case["y_train"])
        y_test = pd.Series(case["y_test"])
        expected = case["expected"]

        # Execute the encode_labels function
        le, y_train_enc, y_test_enc = encode_labels(y_train, y_test)

        # Assertions
        # 1. Check that le is an instance of LabelEncoder
        assert isinstance(
            le, LabelEncoder
        ), f"{case_name}: The encoder is not an instance of LabelEncoder."

        # 2. Check that le.classes_ matches expected classes
        assert (
            list(le.classes_) == expected["le_classes"]
        ), f"{case_name}: Expected classes {expected['le_classes']}, got {list(le.classes_)}"

        # 3. Check that y_train_enc matches expected encoded labels
        assert (
            list(y_train_enc) == expected["y_train_enc"]
        ), f"{case_name}: Expected y_train_enc {expected['y_train_enc']}, got {list(y_train_enc)}"

        # 4. Check that y_test_enc matches expected encoded labels
        assert (
            list(y_test_enc) == expected["y_test_enc"]
        ), f"{case_name}: Expected y_test_enc {expected['y_test_enc']}, got {list(y_test_enc)}"
