import pandas as pd

class EdaAnalyser:
    def __init__(self, df):
        self.df = df
        self.dtypes = {}

    def summarize_overview(self) -> dict:
        '''
        Summarize the overview of the DataFrame.
        '''
        self.classify_columns()
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "missing_values": self.df.isnull().sum().sum(),
            "missing_by_column": self.df.isnull().sum().to_dict()
        }

    def classify_columns(self):
        for col in self.df.columns:
            series = self.df[col]

            if pd.api.types.is_bool_dtype(series):
                self.dtypes[col] = "boolean"
            elif pd.api.types.is_numeric_dtype(series):
                self.dtypes[col] = "numerical"
            elif pd.api.types.is_datetime64_any_dtype(series):
                self.dtypes[col] = "datetime"
            else:
                self.dtypes[col] = "categorical"

    def describe_numeric(self, col: str) -> dict:
        '''
        Summarize the numerical column of the DataFrame.
        '''
        res = {
            "dtype": [self.dtypes[col]],
        }
        if res["dtype"][0] == "numerical":
            res['mean']  = [self.df[col].mean()]
            res['std']   = [self.df[col].std()]
            res['min']   = [self.df[col].min()]
            res['max']   = [self.df[col].max()]
            res['Q1']    = [self.df[col].quantile(0.25)]
            res['Q3']    = [self.df[col].quantile(0.75)]
            res['IQR']   = [self.df[col].quantile(0.75) - self.df[col].quantile(0.25)]
            res['range'] = [self.df[col].max() - self.df[col].min()]
            res['skew']  = [self.df[col].skew()]

        return res
