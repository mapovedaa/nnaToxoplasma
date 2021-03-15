# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy  as np

# Preprocesado y modelado
# ==============================================================================
from   sklearn.linear_model     import LogisticRegression
from   sklearn.model_selection  import train_test_split
from   sklearn.metrics          import accuracy_score
from   sklearn.metrics          import plot_confusion_matrix
import statsmodels.api          as     sm
import statsmodels.formula.api  as     smf

def extractDataCSV(csv,columnName):
    X = csv.drop(columns = columnName).astype(float)
    y = csv[columnName]

    return X,y

def removeAtipicalData(X, rangeMinimun):
    return X[X.columns[X.sum()>rangeMinimun]]

def configCSV(csv):
    csvConfig               = csv.transpose()
    csvConfig.columns       = csvConfig.iloc[0].values
    csvConfig               = csvConfig.drop(csvConfig.index[0])
    csvConfig               = csvConfig.assign(state=csvConfig.index.values)
    csvConfig               = csvConfig.reset_index(drop=True)
    csvConfig["infected"]   = np.where(csvConfig["state"].str.contains("Chronico"), 1, 0)
    csvConfig               = csvConfig.drop(columns=["state"])

    return csvConfig

def readFile(pathFile):
    pathFile  = "/home/mike/Downloads/data.transform5.csv"
    df_comma  = pd.read_csv(pathFile, nrows=1,sep=",")
    df_semi   = pd.read_csv(pathFile, nrows=1, sep=";")
    csv       = pd.DataFrame
    if df_comma.shape[1]>df_semi.shape[1]:
        print("comma delimited")
        csv       = pd.read_csv(pathFile, sep=",")
    else:
        print("semicolon delimited")
        csv       = pd.read_csv(pathFile, sep=";")
    return csv

csv = readFile('/home/mike/Documents/tesis/nnaToxoplasma/data.csv')
dataset = configCSV(csv)

print("Número de observaciones por clase")
print(dataset['infected'].value_counts())
print("#=====================================")
print("Porcentaje de observaciones por clase")
print(100 * dataset['infected'].value_counts(normalize=True))

# División de los datos en train y test
# ==============================================================================
X, y    = extractDataCSV(dataset, 'infected')
X       = removeAtipicalData(X, 100000000000)

X_train, X_test, y_train, y_test = train_test_split(X                     ,
                                                    y.values.reshape(-1,1),
                                                    train_size   = 0.7    ,
                                                    random_state = 1234   ,
                                                    shuffle      = True   )
# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
model   = sm.Logit(endog=y_train, exog=X_train, missing='drop')
print(X_train)
print("=======================")
print(y_train)
model   = model.fit()
print(model.summary())