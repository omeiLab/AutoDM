import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from module.utils import classify_dtype

class MissingValuesHandler:
    def __init__(self, df):
        self.df = df
        self.stat = self._missing_stat()
        self.methods  = {
            'numerical': ['mean', 'median'],
            'categorical': ['mode', 'unknown'],
            'datetime': ['forward fill', 'backward fill', 'interpolate by time']
        }

    def show_methods(self, col):
        dtype = classify_dtype(self.df[col])
        return self.methods[dtype] + ['drop row', 'drop column']

    def _missing_stat(self):
        missing_stats = (
            self.df.isnull().sum()
            .reset_index()
            .rename(columns={'index': 'column', 0: 'missing_count'})
        )
        # drop columns with no missing values
        missing_stats = missing_stats[missing_stats['missing_count'] > 0]
        missing_stats['missing_pct'] = missing_stats['missing_count'] / len(self.df) * 100
        missing_stats['dtype'] = missing_stats['column'].map(lambda col: classify_dtype(self.df[col]))
        # missing_stats = missing_stats.sort_values(by='missing_pct', ascending=False)
        return missing_stats

    def plot_missing_values(self):
        plt.figure(figsize=(10, 6))
        sns.barplot(y='missing_pct', x='column', data=self.stat, hue='dtype', dodge=False)
        plt.xlabel('Missing Values (%)')
        plt.ylabel('Column')
        return plt.gcf()

    def suggest_imputation(self):
        imputation_dict = {}
        for _, row in self.stat.iterrows():
            col = row['column']
            missing_pct = row['missing_pct']
            dtype = row['dtype']
            if missing_pct > 50:
                imputation_dict[col] = "More than 50% missing values, drop the column."
            if missing_pct < 3:
                imputation_dict[col] = "Less than 3% missing values, drop the row."
            elif dtype == 'numerical':
                skewness = self.df[col].skew()
                if skewness > 1:
                    imputation_dict[col] = f"Highly skewed, use median imputation."
                else:
                    imputation_dict[col] = f"Not skewed, use mean imputation."
            elif dtype == 'categorical':
                if missing_pct > 5:
                    imputation_dict[col] = f"Assign 'unknown' as a new category."
                else:
                    imputation_dict[col] = f"Use mode imputation."
            elif dtype == 'datetime':
                imputation_dict[col] = f"Interpolate base on your understanding to the data."
        return imputation_dict

    # not show for "drop"
    def impute_plot_preview(self, col, method):
        dtype = classify_dtype(self.df[col])
        series_before = self.df[col].copy()
        series_after = self.impute(series_before, method)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        # plot by dtype
        if dtype == 'numerical' or dtype == 'datetime':
            sns.histplot(series_before.dropna(), ax=ax[0], kde=True, color='skyblue')
            sns.histplot(series_after, ax=ax[1], kde=True, color='orange')
        elif dtype == 'categorical':
            count_before = series_before.value_counts()
            count_after = series_after.value_counts()
            sns.barplot(x=count_before.index, y=count_before.values, ax=ax[0], color='skyblue')
            sns.barplot(x=count_after.index, y=count_after.values, ax=ax[1], color='orange')
            ax[0].tick_params(axis='x', rotation=45)
            ax[1].tick_params(axis='x', rotation=45)

        ax[0].set_title('Before Imputation')
        ax[1].set_title(f'After Imputation ({method})')
        return fig


    def impute(self, series_before, method):
        if method == 'mean':
            return series_before.fillna(series_before.mean())
        elif method =='median':
            return series_before.fillna(series_before.median())
        elif method =='mode':
            return series_before.fillna(series_before.mode().iloc[0])
        elif method == 'unknown':
            return series_before.fillna('unknown')
        elif method == 'forward fill':
            return series_before.fillna(method='ffill')
        elif method == 'backward fill':
            return series_before.fillna(method='bfill')
        elif method == 'interpolate by time':
            return series_before.interpolate(method='time')
        else:
            return series_before



