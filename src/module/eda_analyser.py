import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

class EdaAnalyser:
    def __init__(self, df):
        self.df = df
        self.dtypes = {}
        self.cast_object()
        self.classify_columns()

    def summarize_overview(self) -> dict:
        '''
        Summarize the overview of the DataFrame.
        '''
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "missing_values": self.df.isnull().sum().sum(),
            "missing_by_column": self.df.isnull().sum().to_dict()
        }

    def cast_object(self):
        '''
        Try to cast every oject column to numeric.
        '''
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.try_cast_numeric(self.df[col])

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

    def classify_columns(self, cat_threshold=10):
        for col in self.df.columns:
            series = self.df[col].dropna()

            if pd.api.types.is_bool_dtype(series):
                self.dtypes[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                self.dtypes[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(series):
                if series.nunique() < cat_threshold:
                    self.dtypes[col] = "categorical"
                else:
                    self.dtypes[col] = "numerical"
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

    def plot_numeric(self, col_name):
        series = self.df[col_name].dropna()

        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # histogram + kde plot
        sns.histplot(series, kde=True, ax=ax[0], color="skyblue")
        ax[0].set_title(f"Distribution of {col_name}")
        
        # boxplot
        sns.boxplot(series, ax=ax[1], color="skyblue")
        ax[1].set_title(f"Boxplot of {col_name}")

        return fig

    def plot_categorical(self, col_name, k = 5):
        series = self.df[col_name].dropna()

        # take top-K
        k = min(k, series.nunique())
        counts = series.value_counts()
        top_k = counts[:k]
        others_sum = counts[k:].sum()
        plot_data = top_k.copy()
        if others_sum > 0:
            plot_data["Others"] = others_sum

        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # count plot
        sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax[0], color="skyblue")
        ax[0].set_title(f"Count of {col_name}")

        # pie plot
        ax[1].pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%', startangle=140)

        return fig