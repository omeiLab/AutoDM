import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from module.EDAnalyser.Univariate.BaseUnivariateAnalyser import BaseAnalyser

class NumericalAnalyser(BaseAnalyser):
    def _validate(self):
        return True
    
    def _get_dtype(self):
        return "numerical"
    
    def _summary(self):
        col = self.col
        report = {
            'mean' : [self.df[col].mean()],
            'std'  : [self.df[col].std()],
            'min'  : [self.df[col].min()],
            'max'  : [self.df[col].max()],
            'Q1'   : [self.df[col].quantile(0.25)],
            'Q3'   : [self.df[col].quantile(0.75)],
            'IQR'  : [self.df[col].quantile(0.75) - self.df[col].quantile(0.25)],
            'range': [self.df[col].max() - self.df[col].min()],
            'skew' : [self.df[col].skew()]
        }
        report = pd.DataFrame(report)
        return report

    def _visualize(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # histogram + kde plot
        sns.histplot(self.series, kde=True, ax=ax[0], color="skyblue")
        ax[0].set_title(f"Distribution of {self.col}")
        
        # boxplot
        sns.boxplot(self.series, ax=ax[1], color="skyblue")
        ax[1].set_title(f"Boxplot of {self.col}")

        plt.tight_layout()
        return fig

