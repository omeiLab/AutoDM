from abc import ABC, abstractmethod
import pandas as pd

class BaseAnalyser(ABC):
    def __init__(self, df, col1, col2, hue):
        self.df = df
        self.col1 = col1
        self.col2 = col2
        self.hue = hue
        self.clean_df = self.make_clean()
        # self.valid = self._validate()

    @abstractmethod
    def _validate(self):
        """
        Check if the data is valid for analysis
        """
        raise NotImplementedError
    
    def make_clean(self):
        series = None
        if self.hue is not None:
            series = self.df[[self.col1, self.col2, self.hue]]
        else:
            series = self.df[[self.col1, self.col2]]
        return series.dropna()

    def analyse(self):
        """
        Perform the univariate analysis
        """
        return {
            "name": self._get_summary_name(),
            "summary": self._summary(),
            "plot": self._visualize()
        }
    
    @abstractmethod
    def _get_summary_name(self):
        """
        Get the name of the summary
        """
        raise NotImplementedError

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
