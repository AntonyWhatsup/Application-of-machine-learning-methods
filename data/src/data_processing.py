import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads the Excel dataset and performs initial structural cleaning."""
    try:
        # header=3 skips the first rows with metadata
        df = pd.read_excel(file_path, header=3)
        # Remove empty columns that pandas automatically named Unnamed
        df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_missing_values(df):
    """Handles missing values by dropping sparse columns and imputing the rest."""
    df_cleaned = df.copy()
    
    # Drop columns with more than 50% missing values
    threshold = len(df_cleaned) * 0.5
    df_cleaned = df_cleaned.dropna(thresh=threshold, axis=1)
    
    # Fill remaining numerical NaNs with median
    num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
    # Fill remaining categorical NaNs with mode
    cat_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        
    return df_cleaned

def engineer_features(df):
    """Generates new categorical and numerical features."""
    df_feat = df.copy()
    
    # Feature 1: Experience Level based on YearsWithThisTypeOfJob
    if 'YearsWithThisTypeOfJob' in df_feat.columns:
        conditions = [
            (df_feat['YearsWithThisTypeOfJob'] <= 2),
            (df_feat['YearsWithThisTypeOfJob'] > 2) & (df_feat['YearsWithThisTypeOfJob'] <= 5),
            (df_feat['YearsWithThisTypeOfJob'] > 5)
        ]
        choices = ['Junior', 'Mid', 'Senior']
        df_feat['Experience_Level'] = np.select(conditions, choices, default='Unknown')

    # Feature 2: Specialization Ratio
    if 'YearsWithThisDatabase' in df_feat.columns and 'YearsWithThisTypeOfJob' in df_feat.columns:
        df_feat['Specialization_Ratio'] = df_feat['YearsWithThisDatabase'] / (df_feat['YearsWithThisTypeOfJob'] + 0.1)
        
    return df_feat