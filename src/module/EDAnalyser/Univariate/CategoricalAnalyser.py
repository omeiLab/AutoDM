import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from module.EDAnalyser.Univariate.BaseUnivariateAnalyser import BaseAnalyser

class CategoricalAnalyser(BaseAnalyser):
    def _validate(self):
        def _is_high_cardinality(self, threshold=0.5):
            return self.series.nunique() / len(self.series) > threshold
        
        return not _is_high_cardinality(self)
    
    def _compress_categories(self, k: int = 10) -> pd.Series:
        """
        Take top-K of categorical variable and replace the rest with "Others"
        """
        top_k = self.series.value_counts().nlargest(k).index
        cat_series = self.series.apply(lambda x: x if x in top_k else "Others")
        return cat_series
    
    def _get_dtype(self):
        return "categorical"
    
    def _summary(self):
        return None

    def _visualize(self):
        if not self._validate():
            return None
        
        plot_data = self._compress_categories().value_counts()
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # histogram + kde plot
        sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax[0], color="skyblue")
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].set_title(f"Count of {self.col}")
        
        # pie plot
        ax[1].pie(plot_data.values, labels=plot_data.index, startangle=140, pctdistance=0.8,)

        plt.tight_layout()
        return fig

