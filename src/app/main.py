import logging
import time
from flask import Flask, request, jsonify
from src.model.inference import SentimentInference

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the inference engine at app startup
inference_engine = SentimentInference()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the sentiment of a given text.

    This endpoint receives a JSON payload containing a "text" field, processes the text,
    and returns the predicted sentiment along with the probabilities for each sentiment class.

    Returns:
        tuple: A Flask response object containing a JSON payload and an HTTP status code.
            - On success (200 OK): {"sentiment": "<predicted_sentiment>"}
            - On error (400 Bad Request): {"error": "No text field provided"}
    """
    start_time = time.time()
    data = request.get_json(force=True, silent=True)

    if not data or "text" not in data:
        logger.error("No 'text' provided in request.")
        return jsonify({"error": "No text field provided"}), 400

    text = data["text"]
    predictions = inference_engine.predict_sentiment(text)

    latency = (time.time() - start_time) * 1000
    logger.info(f"Latency: {latency:.2f} ms")
    logger.info(predictions)

    result = max(predictions, key=predictions.get)

    return jsonify(
        {
            "sentiment": result,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for liveness and readiness.

    This endpoint can be used by monitoring tools or orchestration platforms
    to verify that the application is running and responsive.

    Returns:
        tuple: A Flask response object containing a JSON payload and an HTTP status code.
            - On success (200 OK): {"status": "ok"}
    """
    # Health endpoint for liveness checks
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # Environment variables or config files could determine host/port in production
    app.run(host="0.0.0.0", port=5000)
