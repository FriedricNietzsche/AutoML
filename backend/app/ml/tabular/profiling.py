def profile_dataset(df):
    """
    Profiles the given dataset and returns summary statistics and data types.

    Parameters:
    df (pd.DataFrame): The dataset to profile.

    Returns:
    dict: A dictionary containing summary statistics and data types.
    """
    profile = {
        "columns": {},
        "summary": {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "missing_values": df.isnull().sum().to_dict(),
        }
    }

    for column in df.columns:
        profile["columns"][column] = {
            "dtype": str(df[column].dtype),
            "unique_values": df[column].nunique(),
            "sample_values": df[column].dropna().unique()[:5].tolist()
        }

    return profile


def detect_data_types(df):
    """
    Detects and returns the data types of the columns in the dataset.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary mapping column names to their data types.
    """
    return {col: str(df[col].dtype) for col in df.columns}