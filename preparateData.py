import numpy  as np
import pandas as pd

from   sklearn.feature_selection    import VarianceThreshold
from   sklearn.compose              import ColumnTransformer
from   sklearn.impute               import SimpleImputer
from   sklearn.preprocessing        import StandardScaler
from   sklearn.pipeline             import Pipeline
from   sklearn.model_selection      import train_test_split

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

def main():
    pathCSV = str(input("please, put path file csv (default = data.csv): ") or "data.csv")

    csv = readFile(pathCSV)
    dataset = configCSV(csv)

    X, y    = extractDataCSV(dataset, 'infected')
    X       = removeAtipicalData(X, .8 ,0)

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

    X_train, X_test, y_train, y_test = train_test_split(X_prep                ,
                                                        y.values.reshape(-1,1),
                                                        train_size   = 0.7    ,
                                                        random_state = 1234   ,
                                                        shuffle      = True   )
    X_train.to_csv('./data/training.csv', index=False)
    y_train.to_csv('./data/result_training.csv', index=False)

    X_test.to_csv('./data/testing.csv', index=False)
    y_test.to_csv('./data/result_testing.csv', index=False)

if __name__ == "__main__":
    main()