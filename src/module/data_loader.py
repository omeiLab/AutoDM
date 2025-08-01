import pandas as pd
import streamlit as st

class DataLoader:
    def __init__(self, data):
        self.data = data

    # @st.cache_data
    def load_data(self, file) -> pd.DataFrame:
        '''
        Load data from a CSV file.
        '''
        self.data = pd.read_csv(file)
        self.cast_object()
        return self.data
    
    def cast_object(self):
        '''
        Try to cast every oject column to numeric.
        '''
        for col in self.data.columns:
            if self.data[col].dtype == "object":
                self.data[col] = self.try_cast_numeric(self.data[col])

    def try_cast_numeric(self, series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)  # remove ,
            .str.replace("%", "", regex=False)  # remove %
            .str.replace("$", "", regex=False)  # remove $
        )

        try:
            numeric = pd.to_numeric(cleaned, errors="raise")
            return numeric
        except Exception:
            return series  # return original series if casting fails