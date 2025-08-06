import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

class BivariateAnalyser:
    def __init__(self, df):
        self.df = df
        self.dtypes = {}
        self.classify_columns()

    def classify_columns(self, cat_threshold=20):
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

    def clean_data(self, col1, col2):
        df_clean = self.df[[col1, col2]].dropna()
        return df_clean
    
    def compress_categories(self, cat_series, k: int = 10) -> pd.Series:
        """
        Take top-K of categorical variable and replace the rest with "Others"
        """
        top_k = cat_series.value_counts().nlargest(k).index
        cat_series = cat_series.apply(lambda x: x if x in top_k else "Others")
        return cat_series
    
    def is_high_cardinality(self, series, threshold=0.5):
        return series.nunique() / len(series) > threshold

    def analyse(self, col1, col2, hue_col = None):
        '''
        Status:
            - 1: numerical vs numerical
            - 2: numerical vs categorical
            - 3: categorical vs categorical

            - -1: high cardinality categorical variable

        Dtypes:
            - 0: numerical
            - 1: categorical
            - 2: datetime
        '''
        mapping = {
            "numerical": 0, "categorical": 1, "datetime": 2
        }
        dtype1, dtype2 = mapping[self.dtypes[col1]], mapping[self.dtypes[col2]]

        # numerical vs numerical
        if dtype1 + dtype2 == 0:
            fig, corr = self.num_num_analysis(col1, col2, hue_col)
            return 1, fig, corr
        
        # numerical vs categorical
        elif dtype1 + dtype2 == 1:
            cat = col1 if dtype1 == 1 else col2
            num = col1 if dtype1 == 0 else col2
            if self.is_high_cardinality(self.df[cat]):
                return -1, None, None
            fig, summary_df = self.num_cat_analysis(num, cat, hue_col=hue_col)
            return 2, fig, summary_df
        
        # categorical vs categorical
        elif dtype1 == dtype2 == 1:
            if self.is_high_cardinality(self.df[col1]) or self.is_high_cardinality(self.df[col2]):
                return -1, None, None
            fig, test_result = self.cat_cat_analysis(col1, col2)
            return 3, fig, test_result
        
        # numerical vs datetime
        elif dtype1 + dtype2 == 2:
            num = col1 if dtype1 == 0 else col2
            date = col1 if dtype1 == 2 else col2
            fig, _ = self.num_date_analysis(num, date)
            return 4, fig, None
        

    def num_num_analysis(self, col1: str, col2: str, hue_col = None):
        '''
        Plot numerical vs numerical data & calculate correlation coefficient
        '''
        clean_df = self.df[[col1, col2, hue_col]] if hue_col else self.df[[col1, col2]]
        clean_df = clean_df.dropna()

        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # scatter plot
        sns.scatterplot(x = col1, y = col2, data = clean_df, ax = ax[0], hue = hue_col)
        ax[0].set_title(f'Scatter plot')

        # regplot
        sns.regplot(x = col1, y = col2, data = clean_df, ax = ax[1], scatter_kws = {'alpha': 0.2}, line_kws = {'color': 'r'})
        ax[1].set_title(f'Regression plot')

        # correlation coefficient
        corr_coef = clean_df[col1].corr(clean_df[col2])

        plt.tight_layout()
        return fig, corr_coef
    
    def num_cat_analysis(self, num: str, cat: str, k = 10, hue_col = None):
        '''
        please make sure that col1 is numerical and col2 is categorical
        '''
        clean_df = self.df[[num, cat, hue_col]] if hue_col else self.df[[num, cat]]
        clean_df = clean_df.dropna()

        # take top-K
        clean_df[cat] = self.compress_categories(clean_df[cat], k)

        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # box plot
        sns.boxplot(x = cat, y = num, data = clean_df, ax = ax[0])
        ax[0].set_title(f'Box plot')
        ax[0].tick_params(axis='x', rotation=45)

        # strip plot
        sns.stripplot(x = cat, y = num, data = clean_df, ax = ax[1], jitter = True, hue=hue_col, dodge=True)
        ax[1].set_title(f'Strip plot')
        ax[1].tick_params(axis='x', rotation=45)

        summary_df = (
            clean_df[[num, cat]].dropna()[[cat, num]]
            .dropna()
            .groupby(cat)[num]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .sort_values("count", ascending=False)
        )  

        plt.tight_layout()
        return fig, summary_df

    def cat_cat_analysis(self, col1: str, col2: str):
        clean_df = self.df[[col1, col2]].dropna()

        # take top-K
        clean_df[col1] = self.compress_categories(clean_df[col1])
        clean_df[col2] = self.compress_categories(clean_df[col2])

        # crosstab
        table = pd.crosstab(clean_df[col1], clean_df[col2])
        
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # heatmap 
        sns.heatmap(table, annot=True, fmt='g', ax = ax[0], cmap='Blues')
        ax[0].set_title(f'Heatmap of Contingency Table')

        # statcked bar chart, distribution of each category
        ct_pct = table.div(table.sum(axis=1), axis=0)
        ct_pct.plot(kind='bar', stacked=True, ax=ax[1])
        ax[1].set_title(f'Stacked Bar Chart of Distribution')

        # chi-square test
        test_result = chi2_contingency(table)
        
        plt.tight_layout()
        return fig, test_result
    
    def num_date_analysis(self, num: str, date: str):
        clean_df = self.df[[num, date]].dropna()

        # line plot
        fig, ax = plt.subplots(1, 1, figsize = (12, 6))
        sns.lineplot(x = date, y = num, data = clean_df, ax = ax)
        ax.set_title(f'Line plot')

        return fig, None
