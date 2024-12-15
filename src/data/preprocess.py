import re


def clean_text(text) -> str:
    """
    Clean and preprocess input text for sentiment analysis.

    This function takes a string input and performs the following preprocessing steps:

    1. **Type Checking**: Ensures that the input is a string. If not, returns an empty string.
    2. **HTML Tag Removal**: Replaces HTML `<br>` tags with a space to eliminate line breaks.
    3. **Lowercasing**: Converts all characters in the text to lowercase to ensure uniformity.
    4. **Whitespace Trimming**: Removes leading and trailing whitespace from the text.

    Args:
        text (str): The raw input text to be cleaned and preprocessed.

    Returns:
        str: The cleaned and preprocessed text. Returns an empty string if the input is not a string.
    """
    # Check if input is a string
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Lowercase
    text = text.lower().strip()
    return text


def label_sentiment(rating: float) -> str:
    """
    Assign a sentiment label based on the numerical rating.

    This function categorizes a numerical rating into one of three sentiment labels:
    "positive", "neutral", or "negative". The categorization is based on the following criteria:

    - Ratings **greater than or equal to 4.0** are labeled as "positive".
    - A rating **exactly equal to 3.0** is labeled as "neutral".
    - Ratings **below 3.0** are labeled as "negative".

    Args:
        rating (float): The numerical rating to be labeled. Expected to be on a scale where
                        higher values indicate more positive sentiment.

    Returns:
        str: The sentiment label corresponding to the provided rating. Possible values are:
             - "positive"
             - "neutral"
             - "negative"
    """
    if rating >= 4.0:
        return "positive"
    elif rating == 3.0:
        return "neutral"
    else:
        return "negative"
