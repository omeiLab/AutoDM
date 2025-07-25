import pandas as pd

class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self, file) -> pd.DataFrame:
        '''
        Load data from a CSV file.
        '''
        self.data = pd.read_csv(file)
        return self.data