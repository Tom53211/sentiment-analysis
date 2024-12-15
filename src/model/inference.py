import os
import joblib
from xgboost import XGBClassifier
from src.data.preprocess import clean_text

# Constants
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_xgb_model.json")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


class SentimentInference:
    def __init__(self):
        """
        Initializes the SentimentInference class by loading the model and associated artifacts.

        This constructor performs the following actions:

        1. **Model Loading**: Initializes an XGBoost classifier and loads the pre-trained model from the specified path.
        2. **Vectorizer Loading**: Loads the TF-IDF vectorizer from the specified path using joblib.
        3. **Encoder Loading**: Loads the label encoder from the specified path using joblib.
        """
        # Load model and artifacts
        self.model = XGBClassifier()
        self.model.load_model(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.encoder = joblib.load(ENCODER_PATH)

    def predict_sentiment(self, text: str) -> dict:
        """
        Predict the sentiment probabilities for a given text input.

        This method processes the input text by cleaning and vectorizing it,
        then uses the pre-trained model to predict sentiment probabilities
        for each sentiment class.

        Args:
            text (str): The raw input text for which sentiment prediction is desired.

        Returns:
            dict: A dictionary mapping each sentiment class to its corresponding probability.
                  Example:
                      {
                          "negative": 0.05,
                          "neutral": 0.10,
                          "positive": 0.85
                      }
        """

        # Preprocess text
        cleaned = clean_text(text)
        X_vec = self.vectorizer.transform([cleaned])

        # Get probabilities for all classes
        proba = self.model.predict_proba(X_vec)[0]
        classes = self.encoder.classes_

        # Create a dictionary mapping each class to its probability
        results = {cls: float(prob) for cls, prob in zip(classes, proba)}

        return results

# Example usage
if __name__ == "__main__":
    inference_engine = SentimentInference()
    sample_text = "This book was absolutely great!"
    print(inference_engine.predict_sentiment(sample_text))
