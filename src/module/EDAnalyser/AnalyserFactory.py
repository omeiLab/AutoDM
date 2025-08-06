import pandas as pd
from module.EDAnalyser.Univariate.NumericalAnalyser import NumericalAnalyser
from module.EDAnalyser.Univariate.CategoricalAnalyser import CategoricalAnalyser
from module.EDAnalyser.Univariate.DatetimeAnalyser import DatetimeAnalyser

class AnalyserFactory:
    _cache = {}

    @staticmethod
    def create(df, col_name, cat_threshold = 20):
        if col_name in AnalyserFactory._cache:
            return AnalyserFactory._cache[col_name]
        
        series = df[col_name].dropna()
        if pd.api.types.is_datetime64_any_dtype(series):
            analyser = DatetimeAnalyser(df, col_name)
        elif pd.api.types.is_numeric_dtype(series):
            if series.nunique() < cat_threshold:
                analyser = CategoricalAnalyser(df, col_name)
            else:
                analyser = NumericalAnalyser(df, col_name)
        else:
            analyser = CategoricalAnalyser(df, col_name)
        
        AnalyserFactory._cache[col_name] = analyser
        return analyser