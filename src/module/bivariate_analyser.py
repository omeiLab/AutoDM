import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateAnalyser:
    def __init__(self, df):
        self.df = df
        self.dtypes = {}
        self.classify_columns()

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

    def analyse(self, col1, col2):
        if self.dtypes[col1] == "numerical" and self.dtypes[col2] == "numerical":
            fig, corr = self.numerical_numerical_analysis(col1, col2)
            return fig, corr
        else:
            pass
        

    def numerical_numerical_analysis(self, col1: str, col2: str):
        '''
        Plot numerical vs numerical data & calculate correlation coefficient
        '''
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # scatter plot
        sns.scatterplot(x = col1, y = col2, data = self.df, ax = ax[0])
        ax[0].set_title(f'Scatter plot')

        # regplot
        sns.regplot(x = col1, y = col2, data = self.df, ax = ax[1], scatter_kws = {'alpha': 0.2}, line_kws = {'color': 'r'})
        ax[1].set_title(f'Regression plot')

        # correlation coefficient
        corr_coef = self.df[col1].corr(self.df[col2])


        return fig, corr_coef