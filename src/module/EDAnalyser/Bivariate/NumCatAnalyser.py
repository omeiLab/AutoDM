import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from module.utils import classify_dtype
from module.EDAnalyser.Bivariate.BaseBivariateAnalyser import BaseAnalyser

class NumCatAnalyser(BaseAnalyser):
    def __init__(self, df, col1, col2, hue):
        super().__init__(df, col1, col2, hue)
        self.cat = col1 if classify_dtype(self.df[self.col1]) == 'categorical' else col2
        self.num = col1 if classify_dtype(self.df[self.col1]) == 'numerical' else col2
        print(self.cat, self.num)
        
    def _validate(self):
        def _is_high_cardinality(self, threshold=0.5):
            return self.clean_df[self.cat].nunique() / len(self.clean_df) > threshold
        
        return not _is_high_cardinality(self)
    
    def _compress_categories(self, s, k: int = 10) -> pd.Series:
        """
        Take top-K of categorical variable and replace the rest with "Others"
        """
        top_k = s.value_counts().nlargest(k).index
        cat_series = s.apply(lambda x: x if x in top_k else "Others")
        return cat_series
    
    def _get_summary_name(self):
        return "Groupby Analysis"

    def _summary(self):
        if not self._validate():
            return None
        clean_copy = self.clean_df.copy()
        clean_copy[self.cat] = self._compress_categories(clean_copy[self.cat])
        summary_df = (
            clean_copy[[self.num, self.cat]]
            .dropna()
            .groupby(self.cat)[self.num]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .sort_values("count", ascending=False)
        ).round(4)
        return summary_df

    
    def _visualize(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # box plot
        sns.boxplot(x = self.cat, y = self.num, data = self.clean_df, ax = ax[0])
        ax[0].set_title(f'Box plot')
        ax[0].tick_params(axis='x', rotation=45)

        # strip plot
        sns.stripplot(x = self.cat, y = self.num, data = self.clean_df, ax = ax[1], jitter = True, hue=self.hue, dodge=True)
        ax[1].set_title(f'Strip plot')
        ax[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig