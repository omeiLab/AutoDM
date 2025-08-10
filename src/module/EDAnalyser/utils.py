import pandas as pd

def classify_dtype(series, cat_threshold = 20):
    series = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    elif pd.api.types.is_numeric_dtype(series):
        if series.nunique() < cat_threshold:
            return 'categorical'
        else:
            return 'numerical'
    else:
        # Also include boolean and object types as categorical
        return 'categorical'