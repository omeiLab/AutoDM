import pandas as pd
import streamlit as st

class DataLoader:
    def __init__(self, data):
        self.data = data
        self.common_date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', 
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
            '%d.%m.%Y', '%m.%d.%Y',
        ]

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
            if self.data[col].dtype == "object":
                self.data[col] = self.try_cast_datetime(self.data[col])

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
        
    def try_cast_datetime(self, series: pd.Series, threshold = 0.9) -> pd.Series:
        best_series = series
        max_success_ratio = 0.0

        for date_format in self.common_date_formats:
            temp_series = pd.to_datetime(series, format=date_format, errors="coerce")
            success_ratio = temp_series.notna().sum() / len(series)
            if success_ratio > max_success_ratio:
                best_series = temp_series
                max_success_ratio = success_ratio

        return best_series if max_success_ratio > threshold else series