import pandas as pd
import numpy  as np

csv     = pd.read_csv("/home/mike/Documents/tesis/nnaToxoplasma/data.csv")
csvT    = csv.transpose()

print(csvT)
##
features    = csvT.iloc[0].values
csvT        = csvT.drop(csvT.index[0])
attributes  = csvT.index.values
print(attributes)
print(features)
##
csvT.columns      = features
csvT              = csvT.reset_index(drop=True)
csvT              = csvT.assign(state=attributes)
csvT["state"]     = csvT["state"].astype("category")
csvT["infected"]  = np.where(csvT["state"].str.contains("Chronico"), 1, 0)

##
print(csvT.dtypes)
print(csvT)
