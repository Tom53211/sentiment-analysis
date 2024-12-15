# Sentiment Analysis Project

## Project Overview

This project is a **Sentiment Analysis** application that allows users to train a machine learning model to classify text data into sentiments (positive, neutral, negative) and deploy a Flask-based API for real-time sentiment inference. The application is containerized using Docker for easy deployment and includes comprehensive tests to ensure reliability and correctness.

## Features

- **Model Training**: Trains a XGBoost classifier using TF-IDF vectorized text data.
- **API Deployment**: Deploys a Flask API to serve sentiment predictions.
- **Containerization**: Uses Docker for consistent and scalable deployment.
- **Testing**: Has unit and integration tests using Pytest with YAML fixtures.
- **Logging**: Monitord application performance and behavior through logging.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System**: Linux, macOS
- **Python**: Version 3.10 or higher
- **Docker**: Installed and running
- **Poetry**: For dependency management

## Usage

### Training the Model

To train the sentiment analysis model, run the following command:

```bash
poetry run python3 src/model/train.py
```

**Description**: This script processes the raw data, vectorizes the text using TF-IDF, encodes the sentiment labels, and trains an XGBoost classifier. The trained model and associated artifacts are saved in the `models/` directory.

### Deploying the Flask Inference App

To deploy the Flask-based sentiment inference API using Docker, execute the following commands:

1. **Build the Docker Image**

   ```bash
   docker build -f src/app/Dockerfile -t sentiment-app:latest .
   ```

   **Description**: This command builds a Docker image named `sentiment-app` using the `Dockerfile` located in `src/app/`.

2. **Run the Docker Container**

   ```bash
   docker run -p 5000:5000 sentiment-app:latest
   ```

   **Description**: This command runs the Docker container, mapping port `5000` of your local machine to port `5000` of the container, making the API accessible at `http://localhost:5000`.


### Testing the Inference App

You can test the deployed Flask inference API using `curl` commands as follows:

1. **Predict Sentiment**

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"text":"I loved this product!"}' http://127.0.0.1:5000/predict
   ```

   **Expected Response**:

   ```json
   {
     "sentiment": "positive"
   }
   ```

2. **Health Check**

   ```bash
   curl -X GET http://127.0.0.1:5000/health
   ```

   **Expected Response**:

   ```json
   {
     "status": "ok"
   }
   ```

## API Endpoints

### 1. Predict Sentiment

- **URL**: `/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Payload**:

  ```json
  {
    "text": "Your input text here"
  }
  ```

- **Success Response**:
  - **Code**: `200 OK`
  - **Content**:

    ```json
    {
      "sentiment": "positive"
    }
    ```

- **Error Response**:
  - **Code**: `400 Bad Request`
  - **Content**:

    ```json
    {
      "error": "No text field provided"
    }
    ```

### 2. Health Check

- **URL**: `/health`
- **Method**: `GET`
- **Success Response**:
  - **Code**: `200 OK`
  - **Content**:

    ```json
    {
      "status": "ok"
    }
    ```

### Running Tests

To run all tests in the `tests/` directory, use the following command:

```bash
poetry run pytest tests/
```

**Description**: This command executes all test cases using Pytest, ensuring that your model training, vectorization, label encoding, and API endpoints are functioning correctly.