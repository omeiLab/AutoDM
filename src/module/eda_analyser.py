import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

class EdaAnalyser:
    def __init__(self, df):
        self.df = df
        self.dtypes = {}
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

        plt.tight_layout()
        return fig
    
    def is_high_cardinality(self, s, threshold=0.5):
        return s.nunique() / len(s) > threshold

    def plot_categorical(self, col_name, k = 10):
        series = self.df[col_name].dropna()

        if self.is_high_cardinality(series):
            return None

        # take top-K
        if k < series.nunique():
            counts = series.value_counts()
            top_k = counts[:k]
            others_sum = counts[k:].sum()
            plot_data = top_k.copy()
            if others_sum > 0:
                plot_data["Others"] = others_sum
        else:
            plot_data = series.value_counts()

        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # count plot
        sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax[0], color="skyblue")
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].set_title(f"Count of {col_name}")

        # pie plot
        ax[1].pie(plot_data.values, labels=plot_data.index, startangle=140, pctdistance=0.8,)

        plt.tight_layout()
        return fig
    
    def infer_time_granularity(self, col_name):
        granularity = []
        dt = self.df[col_name].dropna()
        
        if dt.dt.second.nunique() > 1:
            granularity.append('second')
        if dt.dt.minute.nunique() > 1:
            granularity.append('minute')
        if dt.dt.hour.nunique() > 1:
            granularity.append('hour')
        if dt.dt.dayofweek.nunique() > 1:
            granularity.append('dayofweek')
        if dt.dt.day.nunique() > 1:
            granularity.append('day')
        if dt.dt.month.nunique() > 1:
            granularity.append('month')
        if dt.dt.year.nunique() > 1:
            granularity.append('year')
        
        return granularity

    def plot_datetime(self, col_name, period):
        series = self.df[col_name].dropna()

        if period == 'year':
            dt = series.dt.year.value_counts().sort_index()
            xlabel = 'Year'
        elif period =='month':
            dt = series.dt.month.value_counts().sort_index()
            xlabel = 'Month'
        elif period == 'day':
            dt = series.dt.day.value_counts().sort_index()
            xlabel = 'Day'
        elif period == 'dayofweek':
            # set 0 - 6 to Monday - Sunday
            dt = series.dt.dayofweek.value_counts().sort_index()
            day_names = {
                0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
            }
            dt.index = dt.index.map(day_names)
            xlabel = 'Day of Week'
        elif period == 'hour':
            dt = series.dt.hour.value_counts().sort_index()
            xlabel = 'Hour'
        elif period =='minute':
            dt = series.dt.minute.value_counts().sort_index()
            xlabel = 'Minute'
        elif period =='second':
            dt = series.dt.second.value_counts().sort_index()
            xlabel = 'Second'

        # count plot
        fig, ax = plt.subplots(1, 1, figsize = (12, 6))
        sns.barplot(x=dt.index, y=dt.values, ax=ax, color="skyblue")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(f"Count of {col_name} by {period}")

        plt.tight_layout()
        return fig