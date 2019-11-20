import numpy as np
import pandas as pd

class data:
    def __init__(self, fileName):
        df = pd.read_csv(fileName)
        values = df.values
        self.stars = np.array(values[:, 0], dtype = int)
        self.names = values[:, 1]
        self.text = values[:, 2]
        self.date = values[:, 3]
        self.city = values[:, 7]
        self.category = values[:, 10]
        self.sentiment = np.nan_to_num(values[:, 13])
        s = {0, 1, 2, 3, 7, 10}
        cols = []
        for i in range(values.shape[1]):
            if i not in s: cols.append(i)
        self.numerical = np.array(values[:, cols], dtype = np.float64)
    
    def center(self):
        mean = np.nanmean(self.numerical, axis = 0)
        self.numerical -= mean
        self.numerical = np.nan_to_num(self.numerical)

