import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from module.utils import classify_dtype

class CategoricalEncodingHandler:
    def __init__(self, df):
        self.df = df
        self.cat_cols = [col for col in df.columns if classify_dtype(df[col]) == "categorical"]
        self.methods = [
            "frequency encoding", "one-hot encoding", 
            "label encoding (ordinal)", "label encoding (nominal)", 
        ]

    def show_methods(self):
        return self.methods

    # return unique counts & missing ratio
    def show_summary(self, col):
        n_unique = self.df[col].nunique()
        n_missing = self.df[col].isnull().sum() / len(self.df)
        return n_unique, n_missing

    def suggest_encoding(self):
        encoding_dict = {}
        for col in self.cat_cols:
            n_unique = self.df[col].nunique()
            if n_unique <= 10:
                encoding_dict[col] = "one-hot encoding"
            elif n_unique <= 50:
                encoding_dict[col] = "frequency encoding"
            else:
                encoding_dict[col] = "label encoding (nominal)"
        return encoding_dict
    
    def _frequency_encoding(self, col, preview = False):
        copy = self.df.copy()
        copy[col] = copy[col].astype(str)
        encoded = copy[col].value_counts().to_dict()
        encoded = {str(k): v for k, v in encoded.items()}
        copy[f'{col}_freq_encode'] = copy[col].map(encoded)
        copy.drop(col, axis=1, inplace=True)

        return copy[[f'{col}_freq_encode']].astype(str) if preview else copy


    def _one_hot_encoding(self, col, preview = False):
        copy = self.df.copy()
        encoded = pd.get_dummies(copy[col], prefix=col)
        copy = pd.concat([copy, encoded], axis=1)
        copy.drop(col, axis=1, inplace=True)
        return copy[list(encoded.columns)].astype(str) if preview else copy

    def _label_encoding_ordinal(self, col, mapping, preview = False):
        copy = self.df.copy()
        copy[col] = copy[col].astype(str)
        copy[f'{col}_label_encode'] = copy[col].map(mapping)
        copy.drop(col, axis=1, inplace=True)
        return copy[[f'{col}_label_encode']].astype(str) if preview else copy
    
    def _label_encoding_nominal(self, col, preview = False):
        copy = self.df.copy()
        encoded = pd.factorize(copy[col])[0]
        copy[f'{col}_label_encode'] = encoded
        copy.drop(col, axis=1, inplace=True)
        return copy[f'{col}_label_encode'].astype(str) if preview else copy
    
    def preview_encoding(self, col, method, mapping=None):
        processed = self.process(col, method, mapping, preview=True)
        return self.df[[col]], processed

    def process(self, col, method, mapping=None, preview=False):
        if method == "frequency encoding":
            processed_df = self._frequency_encoding(col, preview)
        elif method == "one-hot encoding":
            processed_df = self._one_hot_encoding(col, preview)
        elif method == "label encoding (ordinal)":
            processed_df = self._label_encoding_ordinal(col, mapping, preview)
        elif method == "label encoding (nominal)":
            processed_df = self._label_encoding_nominal(col, preview)
        else:
            raise ValueError(f"Invalid method: {method}")
        return processed_df