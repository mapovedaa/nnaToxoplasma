from sklearn.neural_network     import MLPClassifier
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score

import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/' \
    + 'Estadistica-machine-learning-python/master/data/spam.csv'
datos = pd.read_csv(url)

datos['type'] = np.where(datos['type'] == 'spam', 1, 0)

print("Número de observaciones por clase")
print(datos['type'].value_counts())
print("")
print("Porcentaje de observaciones por clase")
print(100 * datos['type'].value_counts(normalize=True))

# División de los datos en train y test
# ==============================================================================
X = datos.drop(columns = 'type')
y = datos['type']

print(X)

X_train, X_test, y_train, y_test = train_test_split(X                     ,
                                                    y.values.reshape(-1,1),
                                                    train_size   = 0.7    ,
                                                    random_state = 1234   ,
                                                    shuffle      = True   )

clf = MLPClassifier(solver='lbfgs'           ,
                    alpha=1e-5               ,
                    hidden_layer_sizes=(15,),
                    random_state=1           )

clf.fit(X_train, y_train)
predict     = clf.predict(X_test)
print(predict)
print([coef.shape for coef in clf.coefs_])
probPredict = clf.predict_proba(X_test)
print(probPredict)

accuracy       = accuracy_score(y_true    = y_test       ,
                                y_pred    = predict      ,
                                normalize = True         )
print("")
print(f"El accuracy de test es: {100*accuracy}%")