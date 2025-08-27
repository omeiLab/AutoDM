import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from module.utils import classify_dtype

class NumericalHandler:
    def __init__(self, df):
        self.df = df
        self.num_cols = [col for col in df.columns if classify_dtype(df[col]) == "numerical"]
        self.scale_method = ['None', 'MinMax', 'Standard', 'Robust']
        self.outlier_method = ['None', 'Percentile', 'IQR', 'Z-Score']
        self.transform_method = ['None', 'Log', 'Sqrt', 'Box-Cox']
        self.poly_method = ['Linear', 'Quadratic', 'Cubic']

    def get_num_cols(self):
        """
        Returns a list of numerical columns in the dataframe.
        """        
        return self.num_cols
    
    def get_scale_method(self):
        """
        Returns a list of scaling methods.
        """        
        return self.scale_method
    
    def get_outlier_method(self):
        """
        Returns a list of outlier handling methods.
        """
        return self.outlier_method
    
    def get_transform_method(self):
        """
        Returns a list of transformation methods.
        """
        return self.transform_method
    
    def get_poly_method(self):
        """
        Returns a list of polynomial methods.
        """
        return self.poly_method
    
    def summary(self, col):
        """
        Returns a summary of the numerical column.
        """
        report = {
            'mean'  : [self.df[col].mean()],
            'std'   : [self.df[col].std()],
            'min'   : [self.df[col].min()],
            'max'   : [self.df[col].max()],
            'median': [self.df[col].median()],
            'Q1'    : [self.df[col].quantile(0.25)],
            'Q3'    : [self.df[col].quantile(0.75)],
            'IQR'   : [self.df[col].quantile(0.75) - self.df[col].quantile(0.25)],
            'range' : [self.df[col].max() - self.df[col].min()],
            'skew'  : [self.df[col].skew()]
        }
        report = pd.DataFrame(report)
        return report
    
    def preview_plot(self, col, scale_method, outlier_method, transform_method, poly_method):
        after = self.process(col, scale_method, outlier_method, transform_method, poly_method, preview=True)
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))
        sns.histplot(self.df[col], ax=ax[0], kde=True)
        sns.histplot(after, ax=ax[1], kde=True)
        ax[0].set_title('Before Processing')
        ax[1].set_title('After Processing')
        return fig
    
    def process(self, col, scale_method, outlier_method, transform_method, poly_method, preview = False):
        """
        Processes the numerical column by applying the specified methods.
        Args:
            col (str): the column name
            scale_method (str): the scaling method
            outlier_method (str): the outlier handling method
            transform_method (str): the transformation method
            poly_method (str): the degree of polynomial. If not 'Linear', this will generate a new column
            preview (bool): whether just to preview the processed column
        """
        copy = self.df.copy()
        processed_col = copy[col]
        
        # Order: outlier --> transform --> polynomial --> scale
        mask = self.outlier(processed_col, outlier_method)
        copy = copy[mask]
        processed_col = copy[col]
        processed_col = self.transform(processed_col, transform_method)
        processed_col = self.polynomial(processed_col, poly_method)
        processed_col = self.scale(processed_col, scale_method)

        if poly_method != 'Linear':
            poly_dict = {'Quadratic': '^2', 'Cubic': '^3'}
            copy[f"{col}{poly_dict[poly_method]}"] = processed_col
        else:
            copy[col] = processed_col
        
        if preview:
            return processed_col
        
        return copy

    def scale(self, col, method):
        """
        Scale the numerical column using the specified method.
        Args:
            col (pd.Series): the column to be scaled
            method (str): the scaling method
        """
        processed_col = col.copy()
        if method == "MinMax":
            min_val = processed_col.min()
            max_val = processed_col.max()
            processed_col = (col - min_val) / (max_val - min_val)
        elif method == "Standard":
            mean_val = processed_col.mean()
            std_val = processed_col.std()
            processed_col = (col - mean_val) / std_val
        elif method == "Robust":
            Q1 = processed_col.quantile(0.25)
            Q3 = processed_col.quantile(0.75)
            processed_col = (col - Q1) / (Q3 - Q1)
        return processed_col
    
    def outlier(self, col, method):
        """
        Handle the outlier by the specified method.
        Args:
            col (pd.Series): the column to be scaled
            method (str): the scaling method
        """
        processed_col = col.copy()
        if method == "Percentile":
            lower_bound = processed_col.quantile(0.05)
            upper_bound = processed_col.quantile(0.95)
            mask = (col >= lower_bound) & (col <= upper_bound)
        elif method == "IQR":
            Q1 = processed_col.quantile(0.25)
            Q3 = processed_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (col >= lower_bound) & (col <= upper_bound)
        elif method == "Z-Score":
            mean_val = processed_col.mean()
            std_val = processed_col.std()
            z_score = (processed_col - mean_val) / std_val
            mask = (z_score >= -3) & (z_score <= 3)
        else:
            mask = pd.Series(True, index=col.index)
        return mask
    
    def transform(self, col, method):
        """
        Transform the numerical column by the specified method.
        Args:
            col (pd.Series): the column to be scaled
            method (str): the scaling method
        """
        processed_col = col.copy()

        # shifting the distribution to make all positive
        min_val = processed_col.min()
        if min_val < 0:
            processed_col += abs(min_val) + 1

        if method == "Log":
            processed_col = np.log(processed_col)
        elif method == "Sqrt":
            processed_col = np.sqrt(processed_col)
        elif method == "Box-Cox":
            pt = PowerTransformer(method='box-cox', standardize=False)
            processed_col = pd.Series(
                pt.fit_transform(processed_col.values.reshape(-1, 1)).flatten(),
                index=processed_col.index
            )
        return processed_col
    
    def polynomial(self, col, method):
        """
        Generate a new column by applying the specified polynomial method.
        Args:
            col (pd.Series): the column to be scaled
            method (str): the degree of polynomial
        """
        processed_col = col.copy()
        if method == "Quadratic":
            processed_col = col ** 2
        elif method == "Cubic":
            processed_col = col ** 3
        return processed_col
