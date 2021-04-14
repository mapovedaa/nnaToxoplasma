# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================

# Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Datos
# ==============================================================================
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
                                                    train_size   = 0.8    ,
                                                    random_state = 1234   ,
                                                    shuffle      = True   )

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.Logit(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

# ==============================================================================
predicciones = modelo.predict(exog = X_train)

# Clasificación predicha
# ==============================================================================
clasificacion = np.where(predicciones<0.5, 0, 1)
print(clasificacion)

# Accuracy de test del modelo
# ==============================================================================
X_test          = sm.add_constant(X_test, prepend=True)
predicciones    = modelo.predict(exog = X_test)
clasificacion   = np.where(predicciones<0.5, 0, 1)
accuracy        = accuracy_score(y_true    = y_test       ,
                                 y_pred    = clasificacion,
                                 normalize = True         )
print("")
print(f"El accuracy de test es: {100*accuracy}%")

# Matriz de confusión de las predicciones de test
# ==============================================================================
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    clasificacion,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)