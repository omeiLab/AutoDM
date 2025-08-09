import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from module.EDAnalyser.Univariate.BaseUnivariateAnalyser import BaseAnalyser

class DatetimeAnalyser(BaseAnalyser):
    def __init__(self, df, col):
        super().__init__(df, col)
        self.granularity = self._infer_time_granularity()

    def _validate(self):
        return True
    
    def _get_dtype(self):
        return "datetime"
    
    def _summary(self):
        return None
    
    def _infer_time_granularity(self):
        granularity = []
        dt = self.df[self.col].dropna()
        
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
    
    def _get_plot_data(self, period):
        if period == 'year':
            dt = self.series.dt.year.value_counts().sort_index()
            xlabel = 'Year'
        elif period =='month':
            dt = self.series.dt.month.value_counts().sort_index()
            xlabel = 'Month'
        elif period == 'day':
            dt = self.series.dt.day.value_counts().sort_index()
            xlabel = 'Day'
        elif period == 'dayofweek':
            # set 0 - 6 to Monday - Sunday
            dt = self.series.dt.dayofweek.value_counts().sort_index()
            day_names = {
                0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
            }
            dt.index = dt.index.map(day_names)
            xlabel = 'Day of Week'
        elif period == 'hour':
            dt = self.series.dt.hour.value_counts().sort_index()
            xlabel = 'Hour'
        elif period =='minute':
            dt = self.series.dt.minute.value_counts().sort_index()
            xlabel = 'Minute'
        elif period =='second':
            dt = self.series.dt.second.value_counts().sort_index()
            xlabel = 'Second'

        return dt, xlabel # type: ignore
    
    def _visualize(self):
        return None
    
    def visualize_by_period(self, period = None):
        if period is None:
            return ValueError("Please specify the period to visualize.")
        
        dt, xlabel = self._get_plot_data(period)
        fig, ax = plt.subplots(1, 1, figsize = (12, 6))

        # count plot
        sns.barplot(x=dt.index, y=dt.values, ax=ax, color="skyblue")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(f"Count of {self.col} by {period}")

        plt.tight_layout()
        return fig

