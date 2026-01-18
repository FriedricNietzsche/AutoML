"""
Text preprocessing utilities for NLP tasks
Converts text data into numerical features for sklearn models
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text_features(df: pd.DataFrame, text_column: str = "text", max_features: int = 100) -> pd.DataFrame:
    """
    Convert text column into numerical features using TF-IDF vectorization
    
    Args:
        df: DataFrame with text column
        text_column: Name of the text column
        max_features: Maximum number of TF-IDF features to extract
        
    Returns:
        DataFrame with TF-IDF features instead of raw text
    """
    print(f"[TextPreprocessing] Extracting features from '{text_column}' column...")
    print(f"[TextPreprocessing] Input shape: {df.shape}")
    
    if text_column not in df.columns:
        print(f"[TextPreprocessing] ⚠️  Column '{text_column}' not found, skipping text preprocessing")
        return df
    
    # Extract text data
    texts = df[text_column].astype(str).fillna("")
    
    # Apply TF-IDF vectorization
    print(f"[TextPreprocessing] Applying TF-IDF vectorization (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8,  # Ignore terms that appear in more than 80% of documents
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Create DataFrame from TF-IDF features
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{name}" for name in feature_names],
        index=df.index
    )
    
    print(f"[TextPreprocessing] ✅ Created {len(feature_names)} TF-IDF features")
    
    # Add basic text statistics as additional features
    print(f"[TextPreprocessing] Adding basic text statistics...")
    df['text_length'] = texts.str.len()
    df['word_count'] = texts.str.split().str.len()
    df['avg_word_length'] = texts.apply(lambda x: np.mean([len(word) for word in str(x).split()]) if x else 0)
    df['exclamation_count'] = texts.str.count('!')
    df['question_count'] = texts.str.count('\?')
    df['uppercase_ratio'] = texts.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    
    # Drop original text column
    df = df.drop(columns=[text_column])
    
    # Concatenate TF-IDF features with other features
    df = pd.concat([df, tfidf_df], axis=1)
    
    print(f"[TextPreprocessing] ✅ Final shape: {df.shape}")
    print(f"[TextPreprocessing] Features: {list(df.columns)[:10]}... (showing first 10)")
    
    return df


def detect_text_columns(df: pd.DataFrame, max_unique_ratio: float = 0.9) -> list:
    """
    Detect which columns contain text data (high cardinality string columns)
    
    Args:
        df: Input DataFrame
        max_unique_ratio: If unique_values/total_rows > this, consider it text
        
    Returns:
        List of column names that appear to contain text data
    """
    text_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':  # String column
            unique_ratio = df[col].nunique() / len(df)
            avg_length = df[col].astype(str).str.len().mean()
            
            # If most values are unique AND average length > 50 chars, it's likely text
            if unique_ratio > max_unique_ratio and avg_length > 50:
                text_cols.append(col)
                print(f"[TextPreprocessing] Detected text column: '{col}' (unique_ratio={unique_ratio:.2f}, avg_len={avg_length:.0f})")
    
    return text_cols
