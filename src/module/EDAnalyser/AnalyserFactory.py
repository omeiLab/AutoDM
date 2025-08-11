import pandas as pd
from module.utils import classify_dtype
from module.EDAnalyser.Univariate.NumericalAnalyser import NumericalAnalyser
from module.EDAnalyser.Univariate.CategoricalAnalyser import CategoricalAnalyser
from module.EDAnalyser.Univariate.DatetimeAnalyser import DatetimeAnalyser
from module.EDAnalyser.Bivariate.NumNumAnalyser import NumNumAnalyser
from module.EDAnalyser.Bivariate.NumCatAnalyser import NumCatAnalyser
from module.EDAnalyser.Bivariate.CatCatAnalyser import CatCatAnalyser
from module.EDAnalyser.Bivariate.NumTimeAnalyser import NumTimeAnalyser
from module.EDAnalyser.Bivariate.CatTimeAnalyser import CatTimeAnalyser

class AnalyserFactory:
    _cache = {}

    @staticmethod
    def create(df, col_name):
        if col_name in AnalyserFactory._cache:
            return AnalyserFactory._cache[col_name]
        
        dtype = classify_dtype(df[col_name])
        analyser = None
        if dtype == 'datetime':
            analyser = DatetimeAnalyser(df, col_name)
        elif dtype == 'numerical':
            analyser = NumericalAnalyser(df, col_name)
        elif dtype == 'categorical':
            analyser = CategoricalAnalyser(df, col_name)
        
        AnalyserFactory._cache[col_name] = analyser
        return analyser
    
class BivariateAnalyserFactory:
    _cache = {}

    @staticmethod
    def create(df, col1, col2, hue=None):
        if (col1, col2, hue) in BivariateAnalyserFactory._cache:
            return BivariateAnalyserFactory._cache[(col1, col2, hue)]
        if (col2, col1, hue) in BivariateAnalyserFactory._cache:
            return BivariateAnalyserFactory._cache[(col2, col1, hue)]
        
        dtype1 = classify_dtype(df[col1])
        dtype2 = classify_dtype(df[col2])
        analyser = None
        if dtype1 == 'numerical' and dtype2 == 'numerical':
            analyser = NumNumAnalyser(df, col1, col2, hue)
        elif (dtype1 == 'numerical' and dtype2 == 'categorical') or (dtype1 == 'categorical' and dtype2 == 'numerical'):
            analyser = NumCatAnalyser(df, col1, col2, hue)
        elif dtype1 == 'categorical' and dtype2 == 'categorical':
            analyser = CatCatAnalyser(df, col1, col2, hue)
        elif (dtype1 == 'numerical' and dtype2 == 'datetime') or (dtype1 == 'datetime' and dtype2 == 'numerical'):
            analyser = NumTimeAnalyser(df, col1, col2, None)
        elif (dtype1 == 'categorical' and dtype2 == 'datetime') or (dtype1 == 'datetime' and dtype2 == 'categorical'):
            analyser = CatTimeAnalyser(df, col1, col2, None)
        
        BivariateAnalyserFactory._cache[(col1, col2, hue)] = analyser
        return analyser
