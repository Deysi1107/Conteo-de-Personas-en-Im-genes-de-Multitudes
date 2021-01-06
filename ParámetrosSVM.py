# Cargo las librerías necesarias
import datetime
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from os import scandir, getcwd
from os.path import abspath
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.svm._libsvm import predict
from tqdm import tqdm
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import GridSearchCV



 #-----------------------------------
# Mejoras 3. Evaluación de parámetros SVM
# -----------------------------------

print ('Inicia proceso...')
start=datetime.datetime.now()
def grid_search(classifier, X_train, y_train, gparams,
                score='accuracy', cv=10, jobs=-1):
    """
    Realizo una búsqueda de hiperparametros utilizando una grilla
    con la función GridSearchCV de sklearn.model_selection

    Parámetros:
    classifier -- sklearn ML algorithm
    X_train --    Training features
    y_train --    Training labels
    gparams --    Dictionary of parameters to be screened
    score --      Scoring metric
    cv --         K value for Cross Validation
    jobs --       Cores (-1 for all)

    Retorna:
        Mejor modelo
    """


    # Inicializo la clase de GridSearch
    gd_sr=GridSearchCV(estimator=classifier,
                       param_grid=gparams,
                       scoring=score,
                       cv=cv,
                       n_jobs=jobs)

    # Hago un fit de los datos de train
    gd_sr.fit(X_train, y_train)

    # Resultados
    best_result=gd_sr.best_score_
    print("Mejor Resultado = {}".format(best_result))

    # Get the best parameters
    best_parameters=gd_sr.best_params_
    print("Mejores parámetros = {}".format(best_parameters))

    return gd_sr


# Defino los parámetros de búsqueda
params={'C': [0.01, 1, 10], 'gamma': [0.01, 1, 10], 'kernel': ['linear', 'rbf', 'poly']
        }

results=grid_search(SVC(), np.load('allData_HOG.npy'), np.load('allClasses_HOG.npy'), params, score='accuracy', cv=5, jobs=6)

# Otengo todos los resultados de grid search y armo una tabla

cv_results = results.cv_results_
scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
scores_grid_search = scores_df[['params', 'mean_test_score', 'std_test_score']]
print(scores_grid_search)
scores_df.to_csv("HOG_hiperparametros.csv", index=False)
# Guardo los resultados de gridsearch
dump(results, 'results_gridsearch_HOG.pkl')

scores = cross_val_score(SVC(kernel="rbf", C = 10, gamma = 0.01),np.load('allData_HOG.npy'), np.load('allClasses_HOG.npy'), cv=10, n_jobs = 6)

# Con esto puede calcular la exactitud promedio y la varianza
print("Exactitud Promedio (Varianza): %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Muestro la exactitud obtenida en cada fold de 10 Fold CV
print("Exactitud de cada fold= {}".format(scores))
# Creo una SVM con kernel rbf
clf = SVC(kernel="rbf", C = 10, gamma = 0.01, probability = True)

# Entreno la SVM
clf.fit(np.load('allData_HOG.npy'), np.load('allClasses_HOG.npy'))

# Guardo el modelo entreando
dump(clf, 'HOGbest_clf.joblib')
print ( "[INFO] Proceso tomó: {} s" . format(( datetime.datetime. now () - start ) . total_seconds ()))