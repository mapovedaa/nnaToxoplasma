import pandas as pd
import numpy  as np

from   sklearn.feature_selection    import VarianceThreshold
from   sklearn.model_selection      import train_test_split
from   sklearn.metrics              import accuracy_score
from   sklearn.metrics              import plot_confusion_matrix
from   sklearn.metrics              import mean_squared_error
from   sklearn.neural_network       import MLPClassifier
from   sklearn.impute               import SimpleImputer
from   sklearn.preprocessing        import StandardScaler
from   sklearn.compose              import ColumnTransformer
from   sklearn.pipeline             import Pipeline


def extractDataCSV(csv,columnName):
    X = csv.drop(columns = columnName).astype(float)
    y = csv[columnName]

    return X,y

def removeAtipicalData(X, umbral,exp_min_gen):
    sel = VarianceThreshold(threshold=(umbral * (1 - umbral)))
    sel.fit_transform(X)
    return X[X.columns[sel.get_support(indices=True)]]

def configCSV(csv):
    csvConfig               = csv.transpose()
    csvConfig.columns       = csvConfig.iloc[0].values
    csvConfig               = csvConfig.drop(csvConfig.index[0])
    csvConfig               = csvConfig.astype(float)
    csvConfig               = csvConfig.assign(state=csvConfig.index.values)
    csvConfig               = csvConfig.reset_index(drop=True)
    csvConfig["infected"]   = np.where(csvConfig["state"].str.contains("Chronico"), 1, 0)
    csvConfig               = csvConfig.drop(columns=["state"])

    return csvConfig

def readFile(pathFile):
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

def tidy_corr_matrix(corr_mat):
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])

    return(corr_mat)

def multiLayerPerceptron(X_test, y_test, X_train, y_train, lr, alpha, hls):
    clf = MLPClassifier(solver              = 'adam',
                        learning_rate_init  = lr   ,
                        alpha               = alpha,
                        hidden_layer_sizes  = hls  ,
                        verbose             = True ,
                        n_iter_no_change    = 1000 ,
                        max_iter            = 1000 ,
                        random_state        = 1    )
    clf = clf.fit(X_train, y_train)
    clf.predict_proba(X_test)
    # Accuracy de test del modelo
    # ==============================================================================
    predicTest = clf.predict(X_test)
    accuracy = 100*clf.score(X_test, y_test)
    print(f"El accuracy de test es: {accuracy}%")

    confusion_matrix = pd.crosstab( y_test.ravel()         ,
                                    predicTest              ,
                                    rownames=['Real']      ,
                                    colnames=['Predicción'])
    return accuracy

def main():
    pathCSV = str(input("please, put path file csv (default = data.csv): ") or "data.csv")
    csv = readFile(pathCSV)
    dataset = configCSV(csv)

    X, y    = extractDataCSV(dataset, 'infected')
    X       = removeAtipicalData(X, .8 ,0)

    #analisisExploratorio(dataset)
    #distVarNumericas(dataset, ['float64', 'int'])

    #corr_matrix = dataset.select_dtypes(include=['float64', 'int']).corr(method='pearson')
    #heatmapCorrMatrix(corr_matrix)

    # Transformaciones para las variables numéricas
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns.to_list()
    numeric_transformer = Pipeline(
                            steps=[ ('imputer', SimpleImputer(strategy='median')),
                                    ('scaler',  StandardScaler()                )])

    preprocessor = ColumnTransformer(
                        transformers=[
                            ('numeric', numeric_transformer, numeric_cols),
                        ],
                        remainder='passthrough'
                    )

    X_prep = preprocessor.fit_transform(X)
    labels = np.concatenate([numeric_cols])
    X_prep = preprocessor.transform(X)
    X_prep = pd.DataFrame(X_prep, columns=labels)
    # División de los datos en train y test
    # ==============================================================================
    X_train, X_test, y_train, y_test = train_test_split(X_prep                ,
                                                        y.values.reshape(-1,1),
                                                        train_size   = 0.7    ,
                                                        random_state = 1234   ,
                                                        shuffle      = True   )

    _ = multiLayerPerceptron(X_test, y_test, X_train, y_train, 1e-2, 1e-3, (45,12,6,3,))

if __name__ == "__main__":
    main()