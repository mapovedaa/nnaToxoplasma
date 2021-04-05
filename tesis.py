# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy  as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as     plt
import matplotlib.ticker as     ticker
from   matplotlib        import style
import seaborn           as     sns

# Preprocesado y modelado
# ==============================================================================
from   sklearn.feature_selection    import VarianceThreshold
from   sklearn.model_selection      import train_test_split
from   sklearn.metrics              import accuracy_score
from   sklearn.metrics              import plot_confusion_matrix
from   sklearn.metrics              import mean_squared_error
from   sklearn.linear_model         import LinearRegression
from   sklearn.linear_model         import LogisticRegression
from   sklearn.linear_model         import Ridge
from   sklearn.linear_model         import Lasso
from   sklearn.linear_model         import RidgeCV
from   sklearn.linear_model         import LassoCV
from   sklearn.neural_network       import MLPClassifier
from   sklearn.impute               import SimpleImputer
from   sklearn.preprocessing        import StandardScaler
from   sklearn.compose              import ColumnTransformer
from   sklearn.pipeline             import Pipeline
import statsmodels.api          as     sm
import statsmodels.formula.api  as     smf

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
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

def heatmapCorrMatrix(corr_matrix):
    # Heatmap matriz de correlaciones
    # ========================================================
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    sns.heatmap(
        corr_matrix,
        annot     = True,
        cbar      = False,
        annot_kws = {"size": 6},
        vmin      = -1,
        vmax      = 1,
        center    = 0,
        cmap      = sns.diverging_palette(20, 220, n=200),
        square    = True,
        ax        = ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 45,
        horizontalalignment = 'right',
    )

    ax.tick_params(labelsize = 8)
    plt.show()

    return corr_matrix
def distVarNumericas(dataset, includes):
    # Gráfico de distribución para cada variable numérica
    # ================================================================
    # Ajustar número de subplots en función del número de columnas
    columnas_numeric = dataset.select_dtypes(include=includes).columns
    columnas_numeric = columnas_numeric.drop('infected')

    nrows = round(len(columnas_numeric)/5)+1


    fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(9, 5))
    axes = axes.flat

    for i, colum in enumerate(columnas_numeric):
        sns.histplot(
            data    = dataset,
            x       = colum,
            stat    = "count",
            kde     = True,
            color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
            line_kws= {'linewidth': 2},
            alpha   = 0.3,
            ax      = axes[i]
        )
        axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
        axes[i].tick_params(labelsize = 6)
        axes[i].set_xlabel("")

    fig.tight_layout()
    plt.subplots_adjust(top = 0.9)
    fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold")

    plt.show()

    # Se eliminan los axes vacíos
    for i in [nrows]:
        fig.delaxes(axes[i])

    for i, colum in enumerate(columnas_numeric):
        sns.regplot(
            x           = dataset[colum],
            y           = dataset['infected'],
            color       = "gray",
            marker      = '.',
            scatter_kws = {"alpha":0.4},
            line_kws    = {"color":"r","alpha":0.7},
            ax          = axes[i]
        )
        axes[i].set_title(f"infected vs {colum}", fontsize = 7, fontweight = "bold")
        axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
        axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
        axes[i].tick_params(labelsize = 6)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Correlación con precio', fontsize = 10, fontweight = "bold")
    plt.show()

    return dataset
#===========================================================================
def analisisExploratorio(dataset):
    print(dataset.isna().sum().sort_values())
    print("Número de observaciones por clase")
    print(dataset['infected'].value_counts())
    print("#=====================================")
    print("Porcentaje de observaciones por clase")
    print(100 * dataset['infected'].value_counts(normalize=True))

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
    sns.distplot(
        dataset.infected,
        hist    = False,
        rug     = True,
        color   = "blue",
        kde_kws = {'shade': True, 'linewidth': 1},
        ax      = axes[0]
    )
    axes[0].set_title("Distribución original", fontsize = 'medium')
    axes[0].set_xlabel('Infectados', fontsize='small')
    axes[0].tick_params(labelsize = 6)

    fig.tight_layout()
    plt.show()

    print(dataset.select_dtypes(include=['float64', 'int']).describe())
    return dataset

def valorOptimoAlpha(type, X, y, X_test, y_test, X_train, y_train):
    # Creación y entrenamiento del modelo (con búsqueda por CV del valor óptimo alpha)
    # ==============================================================================
    if (type == 'Ridge'):
        # Por defecto RidgeCV utiliza el mean squared error
        model = RidgeCV(alphas          = np.logspace(-10, 2, 200),
                        fit_intercept   = True                    ,
                        normalize       = True                    ,
                        store_cv_values = True                    )
    elif (type == 'Lasso'):
        # Por defecto LassoCV utiliza el mean squared error
        model = LassoCV(alphas      = np.logspace(-10, 3, 200),
                        normalize   = True                    ,
                        cv          = 10                      )

    _ = model.fit(X = X_train, y = y_train)
    # Evolución de los coeficientes en función de alpha
    # ==============================================================================
    alphas = model.alphas
    coefs = []

    for alpha in alphas:
        if (type == 'Ridge'):
            model_tmp = Ridge(alpha=alpha, fit_intercept=False, normalize=True)
        elif (type == 'Lasso'):
            model_tmp = Lasso(alpha=alpha, fit_intercept=False, normalize=True)

        model_tmp.fit(X_train, y_train)
        coefs.append(model_tmp.coef_.flatten())

    fig, ax = plt.subplots(figsize=(7, 3.84))
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    if (type == 'Lasso'):
        ax.set_ylim([-15,None])
    ax.set_xlabel('alpha')
    ax.set_ylabel('coeficientes')
    ax.set_title('Coeficientes del modelo en función de la regularización');
    plt.axis('tight')
    plt.show()

    if (type == 'Lasso'):
        # Número de predictores incluidos (coeficiente !=0) en función de alpha
        # ==============================================================================
        alphas = model.alphas_
        n_predictores = []

        for alpha in alphas:
            modelo_temp = Lasso(alpha=alpha, fit_intercept=False, normalize=True)
            modelo_temp.fit(X_train, y_train)
            coef_no_cero = np.sum(modelo_temp.coef_.flatten() != 0)
            n_predictores.append(coef_no_cero)

        fig, ax = plt.subplots(figsize=(7, 3.84))
        ax.plot(alphas, n_predictores)
        ax.set_xscale('log')
        ax.set_ylim([-15,None])
        ax.set_xlabel('alpha')
        ax.set_ylabel('nº predictores')
        ax.set_title('Predictores incluidos en función de la regularización')
        plt.show()

    # Evolución del error en función de alpha
    # ==============================================================================
    if(type == 'Ridge'):
        # modelo.cv_values almacena el mse de cv para cada valor de alpha. Tiene
        # dimensiones (n_samples, n_targets, n_alphas)
        mse_cv = model.cv_values_.reshape((-1, 200)).mean(axis=0)
        mse_sd = model.cv_values_.reshape((-1, 200)).std(axis=0)
    elif(type == 'Lasso'):
        # modelo.mse_path_ almacena el mse de cv para cada valor de alpha. Tiene
        # dimensiones (n_alphas, n_folds)
        mse_cv = model.mse_path_.mean(axis=1)
        mse_sd = model.mse_path_.std(axis=1)

    # Se aplica la raíz cuadrada para pasar de mse a rmse
    rmse_cv = np.sqrt(mse_cv)
    rmse_sd = np.sqrt(mse_sd)

    # Se identifica el óptimo y el óptimo + 1std
    min_rmse     = np.min(rmse_cv)
    sd_min_rmse  = rmse_sd[np.argmin(rmse_cv)]
    min_rsme_1sd = np.max(rmse_cv[rmse_cv <= min_rmse + sd_min_rmse])
    optimo       = model.alphas[np.argmin(rmse_cv)]
    optimo_1sd   = model.alphas[rmse_cv == min_rsme_1sd]


    # Gráfico del error +- 1 desviación estándar
    fig, ax = plt.subplots(figsize=(7, 3.84))
    ax.plot(model.alphas, rmse_cv)
    ax.fill_between(
        model.alphas,
        rmse_cv + rmse_sd,
        rmse_cv - rmse_sd,
        alpha=0.2
    )

    ax.axvline(
        x         = optimo,
        c         = "gray",
        linestyle = '--',
        label     = 'óptimo'
    )

    ax.axvline(
        x         = optimo_1sd,
        c         = "blue",
        linestyle = '--',
        label     = 'óptimo_1sd'
    )
    ax.set_xscale('log')
    ax.set_ylim([0,None])
    ax.set_title('Evolución del error CV en función de la regularización')
    ax.set_xlabel('alpha')
    ax.set_ylabel('RMSE')
    plt.legend()
    plt.show()

    # Mejor valor alpha encontrado
    # ==============================================================================
    print(f"Mejor valor de alpha encontrado: {model.alpha_}")

    if(type == 'Lasso'):
        # Mejor valor alpha encontrado + 1sd
        # ==============================================================================
        min_rmse     = np.min(rmse_cv)
        sd_min_rmse  = rmse_sd[np.argmin(rmse_cv)]
        min_rsme_1sd = np.max(rmse_cv[rmse_cv <= min_rmse + sd_min_rmse])
        optimo       = model.alphas_[np.argmin(rmse_cv)]
        optimo_1sd   = model.alphas_[rmse_cv == min_rsme_1sd]

        print(f"Mejor valor de alpha encontrado + 1 desviación estándar: {optimo_1sd}")
        # Mejor modelo alpha óptimo + 1sd
        # ==============================================================================
        model = Lasso(alpha=optimo_1sd, normalize=True)
        model.fit(X_train, y_train)

    # Coeficientes del modelo
    # ==============================================================================
    df_coeficientes = pd.DataFrame({'predictor': X_train.columns,
                                    'coef': model.coef_.flatten()})
    if (type == 'Lasso'):
        # Predictores incluidos en el modelo (coeficiente != 0)
        df_coeficientes[df_coeficientes.coef != 0]

    print(df_coeficientes)
    fig, ax = plt.subplots(figsize=(11, 3.84))
    ax.stem(df_coeficientes.predictor, df_coeficientes.coef, markerfmt=' ')
    plt.xticks(rotation=90, ha='right', size=5)
    ax.set_xlabel('variable')
    ax.set_ylabel('coeficientes')
    ax.set_title('Coeficientes del modelo')

    plt.show()
    # Predicciones test
    # ==============================================================================
    predictTest = model.predict(X=X_test)
    predictTest = predictTest.flatten()
    classifTest   = np.where(predictTest<0.5, 0, 1)
    accuracy      = accuracy_score( y_true    = y_test     ,
                                    y_pred    = classifTest,
                                    normalize = True       )

    print(classifTest)
    print("")
    print(f"El accuracy de test es: {100*accuracy}%")

    # Error de test del modelo
    # ==============================================================================
    print("")
    if (type == 'Ridge'):
        rmse_ridge = mean_squared_error(y_true  = y_test     ,
                                        y_pred  = predictTest,
                                        squared = False      )
        print(f"El error (rmse) de test es: {rmse_ridge}")
    elif (type == 'Lasso'):
        rmse_lasso = mean_squared_error(y_true  = y_test     ,
                                        y_pred  = predictTest,
                                        squared = False      )
        print("")
        print(f"El error (rmse) de test es: {rmse_lasso}")

    return _
def minimosCuadrados(X, y, X_test, y_test, X_train, y_train):
    corr_matrix = X.select_dtypes(include=['float64', 'int']).corr(method='pearson')
    print(tidy_corr_matrix(corr_matrix).head(3))

    # Heatmap matriz de correlaciones
    # ==============================================================================
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    sns.heatmap(corr_matrix     ,
                square    = True,
                ax        = ax  )
    ax.tick_params(labelsize = 3)
    plt.show()
    # Creación y entrenamiento del modelo
    # ==============================================================================
    model = LinearRegression(normalize=True)
    model.fit(X = X_train, y = y_train)
    # Coeficientes del modelo
    # ==============================================================================
    df_coeficientes = pd.DataFrame({'predictor': X_train.columns  ,
                                    'coef': model.coef_.flatten()})

    fig, ax = plt.subplots(figsize=(11, 3.84))
    ax.stem(df_coeficientes.predictor, df_coeficientes.coef, markerfmt=' ')
    plt.xticks(rotation=90, ha='right', size=5)
    ax.set_xlabel('variable')
    ax.set_ylabel('coeficientes')
    ax.set_title('Coeficientes del modelo')

    #plt.show()

    # Predicciones test
    # ==============================================================================
    predictTest   = model.predict(X=X_test)
    predictTest   = predictTest.flatten()
    classifTest   = np.where(predictTest<0.5, 0, 1)
    accuracy      = accuracy_score( y_true    = y_test     ,
                                    y_pred    = classifTest,
                                    normalize = True       )

    print(classifTest)
    print("")
    print(f"El accuracy de test es: {100*accuracy}%")

    #Error de test del modelo
    # ==============================================================================
    rmse_ols = mean_squared_error(  y_true  = y_test      ,
                                    y_pred  = predictTest,
                                    squared = False       )
    print("")
    print(f"El error (rmse) de test es: {rmse_ols}")

    return classifTest, accuracy, rmse_ols

def regresionLogisticaSimple(X, y, X_test, y_test, X_train, y_train):
    # Creación del modelo utilizando matrices como en scikitlearn
    # ==============================================================================
    # A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
    X_train = sm.add_constant(X_train, prepend=True)
    model   = sm.Logit(endog=y_train, exog=X_train)
    print(X_train.head(3))
    model   = model.fit()
    # Predicciones con intervalo de confianza
    # ==============================================================================
    predictions = model.predict(exog = X_train)

    # Clasificación predicha
    # ==============================================================================
    clasificacion = np.where(predictions<0.5, 0, 1)
    print(clasificacion)

    # Accuracy de test del modelo
    # ==============================================================================
    X_test     = sm.add_constant(X_test, prepend=True)
    predicTest = model.predict(exog = X_test)
    clasTest   = np.where(predicTest<0.5, 0, 1)
    accuracy   = accuracy_score(y_true    = y_test  ,
                                y_pred    = clasTest,
                                normalize = True    )
    print("")
    print(f"El accuracy de test es: {100*accuracy}%")

    # Matriz de confusión de las predicciones de test
    # ==============================================================================
    confusion_matrix = pd.crosstab( y_test.ravel()         ,
                                    clasTest               ,
                                    rownames=['Real']      ,
                                    colnames=['Predicción'])
    print(confusion_matrix)
    return accuracy, confusion_matrix

def multiLayerPerceptron(X_test, y_test, X_train, y_train):
    lr      = 1e-2
    alpha   = 1e-3
    hls     = (45,12,6,3,)

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

    return accuracy
#===========================================================================
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
X_train, X_test, y_train, y_test = train_test_split(X_prep                     ,
                                                    y.values.reshape(-1,1),
                                                    train_size   = 0.7    ,
                                                    random_state = 1234   ,
                                                    shuffle      = True   )

#_ = regresionLogisticaSimple(X, y, X_test, y_test, X_train, y_train)
#_ = minimosCuadrados(X, y, X_test, y_test, X_train, y_train)
#_ = valorOptimoAlpha('Ridge', X, y, X_test, y_test, X_train, y_train)
#_ = valorOptimoAlpha('Lasso', X, y, X_test, y_test, X_train, y_train)
_ = multiLayerPerceptron(X_test, y_test, X_train, y_train)


