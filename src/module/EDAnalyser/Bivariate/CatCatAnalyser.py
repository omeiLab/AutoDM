import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency
from module.EDAnalyser.Bivariate.BaseBivariateAnalyser import BaseAnalyser

class CatCatAnalyser(BaseAnalyser):

    # TODO: fix this function
    def _validate(self):
        def _is_high_cardinality(self, threshold=0.5):
            return self.series.nunique() / len(self.series) > threshold
        
        return not _is_high_cardinality(self)
    
    def _compress_categories(self, s, k: int = 10) -> pd.Series:
        """
        Take top-K of categorical variable and replace the rest with "Others"
        """
        top_k = s.value_counts().nlargest(k).index
        cat_series = s.apply(lambda x: x if x in top_k else "Others")
        return cat_series
    
    def _get_contingenncy_table(self):
        clean_copy = self.clean_df.copy()
        clean_copy[self.col1] = self._compress_categories(clean_copy[self.col1])
        clean_copy[self.col2] = self._compress_categories(clean_copy[self.col2])
        return pd.crosstab(clean_copy[self.col1], clean_copy[self.col2])
    
    def _get_summary_name(self):
        return "Chi-square test"

    def _summary(self):
        table = self._get_contingenncy_table()
        test_result = chi2_contingency(table)
        return pd.DataFrame({
            "Test Statistic": [test_result[0]],
            "p-value": [test_result[1]],
            "Degree of Freedom": [test_result[2]]
        }).round(4)

    
    def _visualize(self):
        table = self._get_contingenncy_table()
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        # heatmap 
        sns.heatmap(table, annot=True, fmt='g', ax = ax[0], cmap='Blues')
        ax[0].set_title(f'Heatmap of Contingency Table')

        # statcked bar chart, distribution of each category
        ct_pct = table.div(table.sum(axis=1), axis=0)
        ct_pct.plot(kind='bar', stacked=True, ax=ax[1])
        ax[1].set_title(f'Stacked Bar Chart of Distribution')

        plt.tight_layout()
        return fig