# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import logging

from src.data.load_data import load_data
from src.data.preprocess import clean_text, label_sentiment

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = "data/raw/Books_10k.jsonl"
MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_xgb_model.json")


def load_and_process_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a JSON lines file, create sentiment labels, and clean text.

    Args:
        file_path (str): Path to the raw data file.

    Returns:
        pd.DataFrame: Processed DataFrame with sentiment labels and cleaned text.
    """
    # Load data
    df = load_data(file_path)

    # Create sentiment label
    df["sentiment"] = df["rating"].apply(label_sentiment)

    # Clean combined title + text
    df["clean_text"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df["clean_text"] = df["clean_text"].apply(clean_text)

    logger.info(f"Loaded and processed data from {file_path}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    Split the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The processed DataFrame.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.3.
        random_state (int, optional): Seed used by the random number generator. Defaults to 42.

    Returns:
        tuple: Split data (X_train, X_test, y_train, y_test).
    """
    X = df["clean_text"]
    y = df["sentiment"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_vectors(X_train, X_test, max_features: int = 5000):
    """
    Convert text data into TF-IDF vectors.

    Args:
        X_train (pd.Series): Training text data.
        X_test (pd.Series): Testing text data.
        max_features (int, optional): Maximum number of features for TF-IDF. Defaults to 5000.

    Returns:
        tuple: Vectorizer, X_train_vec, X_test_vec
    """
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform on training data
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform testing data
    X_test_vec = vectorizer.transform(X_test)

    logger.info(f"Created TF-IDF vectors with {max_features} features")
    return vectorizer, X_train_vec, X_test_vec


def encode_labels(y_train, y_test):
    """
    Encode sentiment labels into numerical values.

    Args:
        y_train (pd.Series): Training sentiment labels.
        y_test (pd.Series): Testing sentiment labels.

    Returns:
        tuple: LabelEncoder, y_train_enc, y_test_enc
    """
    # Initialize LabelEncoder
    le = LabelEncoder()

    # Fit and transform on training data
    y_train_enc = le.fit_transform(y_train)

    # Transform testing data
    y_test_enc = le.transform(y_test)

    logger.info("Encoded sentiment labels")
    return le, y_train_enc, y_test_enc


def train_model(X_train_vec, y_train_enc) -> XGBClassifier:
    """
    Train the XGBClassifier model.

    Args:
        X_train_vec (sparse matrix): TF-IDF vectors for training data.
        y_train_enc (np.ndarray): Encoded training labels.

    Returns:
        XGBClassifier: Trained model.
    """
    # Initialize XGBoost model
    model = XGBClassifier(eval_metric="logloss")

    # Train model
    model.fit(X_train_vec, y_train_enc)

    logger.info("Trained XGBoost model")
    return model


def evaluate_model(model: XGBClassifier, X_test_vec, y_test, le: LabelEncoder):
    """
    Evaluate the trained model and print metrics.

    Args:
        model (XGBClassifier): Trained model.
        X_test_vec (sparse matrix): TF-IDF vectors for testing data.
        y_test (pd.Series): True sentiment labels for testing data.
        le (LabelEncoder): Fitted LabelEncoder instance.
    """
    # Predict on testing data
    y_pred_enc = model.predict(X_test_vec)
    y_pred = le.inverse_transform(y_pred_enc)

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"Model accuracy: {accuracy:.2f}")
    logger.info(f"Classification report:{report}")


def save_artifacts(
    model: XGBClassifier,
    vectorizer: TfidfVectorizer,
    le: LabelEncoder,
    model_dir: str = MODEL_DIR,
):
    """
    Save the trained model and preprocessing artifacts.

    Args:
        model (XGBClassifier): Trained model.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        le (LabelEncoder): Fitted LabelEncoder instance.
        model_dir (str, optional): Directory to save artifacts. Defaults to MODEL_DIR.
    """
    # Create directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)

    # Save model and artifacts
    model.save_model(MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(le, ENCODER_PATH)

    logger.info("Model and preprocessing artifacts saved.")


def main():
    """
    Main function to execute the training pipeline.
    """
    # Load and process data
    df = load_and_process_data(RAW_DATA_PATH)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Create TF-IDF vectors
    vectorizer, X_train_vec, X_test_vec = create_vectors(X_train, X_test)

    # Encode labels
    le, y_train_enc, y_test_enc = encode_labels(y_train, y_test)

    # Train model
    model = train_model(X_train_vec, y_train_enc)

    # Evaluate model
    evaluate_model(model, X_test_vec, y_test, le)

    # Save artifacts
    save_artifacts(model, vectorizer, le)


if __name__ == "__main__":
    main()
