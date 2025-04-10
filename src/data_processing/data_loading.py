import pandas as pd
from src.config import RAW_DATA_PATH

def load_data():
    """
    Load dataset from RAW_DATA_PATH with basic validation.

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {RAW_DATA_PATH}. Please check the path.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty. Please provide a valid CSV file.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file. Please check the file format.")
    return df
