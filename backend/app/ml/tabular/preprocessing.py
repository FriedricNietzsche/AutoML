from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame, numeric_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by scaling numeric features and encoding categorical features.

    Parameters:
    - df: pd.DataFrame - The input DataFrame to preprocess.
    - numeric_features: list - List of numeric feature column names.
    - categorical_features: list - List of categorical feature column names.

    Returns:
    - pd.DataFrame - The preprocessed DataFrame.
    """
    
    # Define the preprocessing for numeric and categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform the data
    processed_data = preprocessor.fit_transform(df)

    # Convert the result back to a DataFrame
    processed_df = pd.DataFrame(processed_data)

    return processed_df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame by filling them with the mean for numeric features
    and the mode for categorical features.

    Parameters:
    - df: pd.DataFrame - The input DataFrame to process.

    Returns:
    - pd.DataFrame - The DataFrame with missing values handled.
    """
    
    for column in df.columns:
        if df[column].dtype == np.number:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

    return df