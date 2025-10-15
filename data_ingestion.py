import pandas as pd

def load_data(path):
    """Load data from a CSV file."""
    return pd.read_csv(path, index_col=0)

def preprocess_data(df):
    """Preprocess the data by handling missing values."""
    df = df.dropna()
    df = df.drop_duplicates()

    return df