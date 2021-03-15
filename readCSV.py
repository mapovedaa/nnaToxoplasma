# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
# Gráficos
# ==============================================================================

# Preprocesado y modelado
# ==============================================================================
from   sklearn.linear_model    import LogisticRegression
from   sklearn.model_selection import train_test_split
from   sklearn.metrics         import accuracy_score
from   sklearn.metrics         import plot_confusion_matrix
import statsmodels.api         as     sm
import statsmodels.formula.api as     smf
# ==============================================================================
class readCSV():
    def __init__(self, pathCsv):
        pathCSV  = "/home/mike/Downloads/data.transform5.csv"
        df_comma = pd.read_csv(pathCSV, nrows=1,sep=",")
        df_semi  = pd.read_csv(pathCSV, nrows=1, sep=";")

        if df_comma.shape[1]>df_semi.shape[1]:
            print("comma delimited")
            self.csvT       = pd.read_csv(pathCSV, sep=",").transpose()
        else:
            print("semicolon delimited")
            self.csvT       =  pd.read_csv(pathCSV, sep=";").transpose()


    def getCSVTranspose(self):
        features                = self.csvT.iloc[0].values
        self.csvT               = self.csvT.drop(self.csvT.index[0])
        attributes              = self.csvT.index.values

        self.csvT.columns       = features

        self.csvT               = self.csvT.reset_index(drop=True)
        print(self.csvT)
        self.csvT               = self.csvT.astype(float)
        self.csvT               = self.csvT.assign(state=attributes)
        self.csvT["state"]      = self.csvT["state"].astype("category")
        self.csvT["infected"]   = np.where(self.csvT["state"].str.contains("Chronico"), 1, 0)
        self.csvT               = self.csvT.drop(columns=["state"])

        return self.csvT

pathCSV = str(input('full path to csv : '))

csv     = readCSV(pathCSV)
dataset = csv.getCSVTranspose()

print("Número de observaciones por clase")
print(dataset['infected'].value_counts())
print("#=====================================")
print("Porcentaje de observaciones por clase")
print(100 * dataset['infected'].value_counts(normalize=True))

# División de los datos en train y test
# ==============================================================================
X = dataset.drop(columns = 'infected')
y = dataset['infected']

del dataset

trainSize = int(input('pct size data test (%): '))
trainSize = trainSize/100
X_train, X_test, y_train, y_test = train_test_split(X                       ,
                                                    y.values.reshape(-1,1)  ,
                                                    train_size   = trainSize,
                                                    random_state = 1234     ,
                                                    shuffle      = True     )

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
model   = sm.Logit(endog=y_train, exog=X_train)
model   = model.fit()

print(model.summary())