from abc import ABC, abstractmethod
import pandas as pd

class BaseAnalyser(ABC):
    def __init__(self, df, col):
        self.df = df
        self.col = col
        self.series = self.df[self.col].dropna()
        self.valid = self._validate()

    @abstractmethod
    def _validate(self):
        """
        Check if the data is valid for analysis
        """
        raise NotImplementedError

    def analyse(self, overview):
        """
        Perform the univariate analysis

        The overview (Dict): 
            rows             : int, the number of rows in the dataset
            columns          : int, the number of columns in the dataset
            missing_values   : int, the number of missing values in the dataset
            missing_by_column: Dict, the number of missing values by column
        """
        return {
            "dtype": self._get_dtype(),
            "missing_ratio": self._get_missing_ratio(overview),
            "summary": self._summary(),
            "plot": self._visualize()
        }
    
    @abstractmethod
    def _get_dtype(self):
        """
        Get the data type of the column
        """
        raise NotImplementedError
    
    def _get_missing_ratio(self, overview):
        """
        Get the missing ratio of the column
        """
        return overview["missing_by_column"][self.col] / overview["rows"]

    @abstractmethod
    def _summary(self):
        """
        Generate a summary of the analysis results
        """
        raise NotImplementedError

    @abstractmethod
    def _visualize(self):
        """
        Visualize the analysis results
        """
        raise NotImplementedError
