import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency
from module.EDAnalyser.utils import classify_dtype
from module.EDAnalyser.Bivariate.BaseBivariateAnalyser import BaseAnalyser

class CatTimeAnalyser(BaseAnalyser):
    def __init__(self, df, col1, col2, hue):
        super().__init__(df, col1, col2, hue)
        self.cat = self.col1 if classify_dtype(self.df[self.col1]) == 'categorical' else self.col2
        self.time = self.col1 if classify_dtype(self.df[self.col1]) == 'datetime' else self.col2
        self.granularity = self._infer_time_granularity()

    # TODO: fix this function
    def _validate(self):
        def _is_high_cardinality(self, threshold=0.5):
            return self.series.nunique() / len(self.series) > threshold
        
        return not _is_high_cardinality(self)
    
    def _infer_time_granularity(self):
        granularity = []
        dt = self.df[self.time].dropna()
        
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
    
    def _compress_categories(self, s, k: int = 10) -> pd.Series:
        """
        Take top-K of categorical variable and replace the rest with "Others"
        """
        top_k = s.value_counts().nlargest(k).index
        cat_series = s.apply(lambda x: x if x in top_k else "Others")
        return cat_series
    
    def _get_contingenncy_table(self, period):
        clean_copy = self.clean_df.copy()
        clean_copy[self.cat] = self._compress_categories(clean_copy[self.cat])
        clean_copy = self._get_period_data(clean_copy, period)
        return pd.crosstab(clean_copy[self.cat], clean_copy[period])

    def _get_summary_name(self):
        return "Linear Regression Analysis"

    def _summary(self):
        return None
    
    def _visualize(self):
        return None
    
    def _granuality_binning(self, dt, period):
        cut = None
        if period == 'day':
            # bin to 7 days a group
            bins = [0, 7, 14, 21, 28, 32]
            cut = pd.cut(dt, bins=bins, labels=['1-7', '8-14', '15-21', '22-28', '29-31'])
        elif period == 'houe':
            bins = [-1, 3, 6, 9, 12, 15, 18, 21, 24]
            cut = pd.cut(dt, bins=bins, labels=['0-3', '4-6', '7-9', '10-12', '13-15', '16-18', '19-21', '22-24'])
        elif period =='minute' or period =='second':
            bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
            cut = pd.cut(dt, bins=bins, labels=['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60'])
        return cut

    def _get_period_data(self, clean_copy, period):
        series = clean_copy[self.time]
        if period == 'year':
            clean_copy[period] = series.dt.year
        elif period =='month':
            clean_copy[period] = series.dt.month
        elif period == 'day':
            clean_copy[period] = series.dt.day
            clean_copy[period] = self._granuality_binning(clean_copy[period], 'day')
        elif period == 'dayofweek':
            # set 0 - 6 to Monday - Sunday
            day_names = {
                0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
            }
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            clean_copy[period] = series.dt.dayofweek.map(day_names)
            clean_copy[period] = pd.Categorical(clean_copy[period], categories=days, ordered=True)
        elif period == 'hour':
            clean_copy[period] = series.dt.hour
            clean_copy[period] = self._granuality_binning(clean_copy[period], 'hour')
        elif period =='minute':
            clean_copy[period] = series.dt.minute
            clean_copy[period] = self._granuality_binning(clean_copy[period],'minute')
        elif period =='second':
            clean_copy[period] = series.dt.second
            clean_copy[period] = self._granuality_binning(clean_copy[period],'second')

        return clean_copy
    
    def analyse_by_period(self, period):
        table = self._get_contingenncy_table(period)
        return {
            "summary": self._summary_by_period(table, period),
            "plot": self._visualize_by_period(table, period)
        }
        
    def _summary_by_period(self, table, period):
        test_result = chi2_contingency(table)
        return pd.DataFrame({
            "Test Statistic": [test_result[0]],
            "p-value": [test_result[1]],
            "Degree of Freedom": [test_result[2]]
        }).round(4)

    def _visualize_by_period(self, table, period):
        fig, ax = plt.subplots(1, 1, figsize = (12, 6))
        sns.heatmap(table, annot=True, fmt='g', cmap='Blues', ax=ax)
        ax.set_title(f"{self.cat} by {period}")
        
        return fig
