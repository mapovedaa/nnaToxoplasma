import pandas as pd
import numpy  as np

class CSV():
    def __init__(self, pathCsv):
        self.path       = pathCsv
        self.csvT       = pd.read_csv(self.path).transpose()

    def getCSVTranspose(self):
        features        = self.csvT.iloc[0].values
        self.csvT       = self.csvT.drop(self.csvT.index[0])
        attributes = self.csvT.index.values

        self.csvT.columns      = features
        self.csvT              = self.csvT.reset_index(drop=True)
        self.csvT              = self.csvT.assign(state=attributes)
        self.csvT["state"]     = self.csvT["state"].astype("category")
        self.csvT["infected"]  = np.where(self.csvT["state"].str.contains("Chronico"), 1, 0)
        self.csvT              = self.csvT.drop(columns=["state"])

        return self.csvT

pathCSV = str(input('full path to csv : '))

csv = CSV(pathCSV)
print(csv.getCSVTranspose())
