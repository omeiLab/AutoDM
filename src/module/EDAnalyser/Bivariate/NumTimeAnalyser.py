import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
from module.utils import classify_dtype
from module.EDAnalyser.Bivariate.BaseBivariateAnalyser import BaseAnalyser

class NumTimeAnalyser(BaseAnalyser):
    def __init__(self, df, col1, col2, hue):
        super().__init__(df, col1, col2, hue)
        self.num = self.col1 if classify_dtype(self.df[self.col1]) == 'numerical' else self.col2
        self.time = self.col1 if classify_dtype(self.df[self.col1]) == 'datetime' else self.col2
        self.granularity = self._infer_time_granularity()

    def _validate(self):
        return True
    
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
    
    def _groupby_granularity(self):
        results = []
        for g in self.granularity:
            temp = self.clean_df.groupby(getattr(self.clean_df[self.time].dt, g))[self.num].mean().reset_index()
            temp.columns = [g, "mean_value"]  
            temp["granularity"] = g
            temp.columns = ["time", "mean_value", "granularity"]    # rename the 1st column to time
            results.append(temp)

        final_df = pd.concat(results, ignore_index=True)
        print(final_df.columns)
        return final_df
    
    def _get_summary_name(self):
        return "Linear Regression Analysis"

    def _summary(self):
        time_ordinal = self.clean_df[self.time].map(datetime.datetime.toordinal)
        corr_coef = self.clean_df[self.num].corr(time_ordinal)
        slope, intercept, r_value, p_value, std_err = linregress(self.clean_df[self.num], time_ordinal)
        return pd.DataFrame({
            "Trend Correlation": [corr_coef],
            "Slope": [slope],
            "Intercept": [intercept],
            "R-squared": [r_value**2], # type: ignore
            "P-value": [p_value],
            "Standard Error": [std_err]
        }).round(4)
    
    def _visualize(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # line plot
        sns.lineplot(x = self.time, y = self.num, data = self.clean_df, ax = ax[0])
        ax[0].set_title(f'Line plot')

        # boxplot of seasonality 
        plot_data = self._groupby_granularity()
        sns.boxplot(x = "granularity", y = "mean_value", data = plot_data, ax = ax[1])
        ax[1].set_title(f'Seasonality')

        plt.tight_layout()
        return fig