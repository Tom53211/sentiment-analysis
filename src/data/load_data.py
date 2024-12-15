import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a JSON Lines file into a pandas DataFrame.

    Args:
        file_path (str): The path to the JSON Lines file to be loaded.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the JSON Lines file.
    """
    df = pd.read_json(file_path, lines=True)
    return df
