import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
from module.EDAnalyser.Bivariate.BaseBivariateAnalyser import BaseAnalyser

class NumNumAnalyser(BaseAnalyser):
    def _validate(self):
        return True
    
    def _get_summary_name(self):
        return "Linear Regression Analysis"

    def _summary(self):
        corr_coef = self.clean_df[self.col1].corr(self.clean_df[self.col2])
        slope, intercept, r_value, p_value, std_err = linregress(self.clean_df[self.col1], self.clean_df[self.col2])
        return pd.DataFrame({
            "Correlation Coefficient": [round(corr_coef, 4)],
            "Slope": [slope],
            "Intercept": [intercept],
            "R-squared": [r_value**2], # type: ignore
            "P-value": [p_value],
            "Standard Error": [std_err]
        }).round(4)
    
    def _visualize(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # scatter plot
        sns.scatterplot(x = self.col1, y = self.col2, data = self.clean_df, ax = ax[0], hue = self.hue)
        ax[0].set_title(f'Scatter plot')

        # regplot
        sns.regplot(x = self.col1, y = self.col2, data = self.clean_df, ax = ax[1], scatter_kws = {'alpha': 0.2}, line_kws = {'color': 'r'})
        ax[1].set_title(f'Regression plot')

        plt.tight_layout()
        return fig