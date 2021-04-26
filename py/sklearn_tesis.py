import pandas as pd
import numpy  as np
import os
import psutil
import time

from    sklearn.metrics         import accuracy_score
from    sklearn.metrics         import plot_confusion_matrix
from    sklearn.metrics         import mean_squared_error
from    sklearn.neural_network  import MLPClassifier


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

    confusion_matrix = pd.crosstab( y_test.ravel()         ,
                                    predicTest              ,
                                    rownames=['Real']      ,
                                    colnames=['Predicción'])
    return accuracy, confusion_matrix

def main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    pathTraining        = str(input("please, put path file training csv (default = ../data/training.csv):") or dirname+"/../data/training.csv")
    pathResultTraining  = str(input("please, put path file result training csv (default = ../data/result_training.csv):") or dirname+"/../data/result_training.csv")

    X_train = readFile(pathTraining)
    y_train = readFile(pathResultTraining).astype(float).to_numpy()

    pathTest       = str(input("please, put path file test csv (default = ../data/testing.csv):") or "data/testing.csv")
    pathResultTest = str(input("please, put path file result test csv (default = ../data/result_test.csv):") or "data/result_testing.csv")

    X_test = readFile(pathTest)
    y_test= readFile(pathResultTest).astype(float).to_numpy()

    start = time.time()

    accuracy, confusion_matrix = multiLayerPerceptron(X_test, y_test, X_train, y_train, 1e-2, 1e-3, (45,12,6,3,))

    print(f"El accuracy de test es: {accuracy}%")
    print(confusion_matrix)

    print("--- %s seconds ---" % (time.time() - start))

if __name__ == "__main__":
    main()