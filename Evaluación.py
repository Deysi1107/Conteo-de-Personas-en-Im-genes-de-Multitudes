# Cargo las librerías necesarias
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


PATH_POSITIVE_TEST="Base de datos/Test/Positivos/"
PATH_NEGATIVE_TEST="Base de datos/Test/Negativos/"

svml = load('HOG_clf.joblib')
# -----------------------------------
# Mejoras 1. Evaluación del modelo
# -----------------------------------
def loadTestingData():
    """
    Funcion para leer los datos de testing

    Retorna:
    testingData -- matriz con las instancias de testing
    classes --     vector con las clases de cada instancia
    """

    # Matriz de descriptores
    testingData=np.array([])
    testingDataNeg=np.array([])

    ################################################
    # Casos positivos
    # Obtengo la lista de casos positivos
    listFiles=[abspath(arch.path) for arch in scandir(PATH_POSITIVE_TEST) if arch.is_file()]
    # Itero los archivos
    for i in tqdm(range(len(listFiles))):
        file=listFiles[i]
        # Leo la imagen
        img=cv2.imread(file, cv2.IMREAD_COLOR)
        # Calculo el HOG
        hog=cv2.HOGDescriptor()
        h=hog.compute(img)
        # Lo paso a 1 dimension
        h2=h.ravel()
        # Agrego a testingData
        testingData=np.hstack((testingData, h2))
    print("Leidas " + str(len(listFiles)) + " imágenes de entrenamiento -> positivas")
    # Hago un reshape
    testingData=testingData.reshape((len(listFiles), len(h2)))
    # Genero el vector de clases
    classes=np.ones(len(listFiles))

    ################################################
    # Casos negativos
    # Obtengo la lista de casos positivos
    listFilesNeg=[abspath(arch.path) for arch in scandir(PATH_NEGATIVE_TEST) if arch.is_file()]
    # Itero los archivos
    for i in tqdm(range(len(listFilesNeg))):
        file=listFilesNeg[i]
        # Leo la imagen
        img=cv2.imread(file, cv2.IMREAD_COLOR)
        # Calculo el HOG
        hog=cv2.HOGDescriptor()
        h=hog.compute(img)
        # Lo paso a 1 dimension
        h2=h.ravel()
        # Agrego a trainingData
        testingDataNeg=np.hstack((testingDataNeg, h2))
    print("Leidas " + str(len(listFilesNeg)) + " imágenes de entrenamiento -> negativas")
    # Hago un reshape
    testingDataNeg=testingDataNeg.reshape((len(listFilesNeg), len(h2)))
    # Genero el vector de clases
    classesNeg=np.zeros(len(listFilesNeg))

    # Merge de los datos
    testingData=np.concatenate((testingData, testingDataNeg), axis=0)
    classes=np.concatenate((classes, classesNeg), axis=0)
    return testingData, classes
# Obtengo los datos de test
X_Test, y_test=loadTestingData()
target_names=['background', 'pedestrians']

# Guardo los datos
np.save("X_Test_HOG", X_Test)
np.save("y_test_HOG", y_test)

# Realizo predicciones sobre el dataset de test
predicciones=svml.predict(X_Test)

# Imprimo un reporte de la clasificación
print(classification_report(y_test, predicciones, target_names=target_names))

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Función para graficar una matriz de confusion

    Parámetros:
    cm --      matriz de confusion
    classes -- etiquetas de las clases
    title --   título del grafico
    cmap --    colores a emplear para graficar
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt='d'
    thresh=cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Imprimo un reporte de la clasificación
print(classification_report(y_test, predicciones, target_names=target_names))

# Calculo la matriz de confusion
cnf_matrix = confusion_matrix(y_test, predicciones)

# Grafico la matriz de confusion
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,title='Matriz de confusion')
plt.show()
plt.savefig("img/Matriz_de_Confusion.png")

# Mejoras 2. K-Fold Cross Validation
# -----------------------------------
# Hago un merge de los datos de train y test
allData = np.concatenate((np.load('X_Train_HOG.npy'), X_Test), axis=0)
allClasses = np.concatenate((np.load('y_train_HOG.npy'), y_test))

# Guardo los datos mergeados
np.save("allData_HOG", allData)
np.save("allClasses_HOG", allClasses)

#allData = np.load("allData_HOG.npy")
#allClasses = np.load("allClasses_HOG.npy")

# Ahora implemento 10 Fold Cross Validation
scores = cross_val_score(SVC(kernel="linear"), allData, allClasses, cv=10, n_jobs = 4)

# Con esto puede calcular la exactitud promedio y la varianza
print("Exactitud Promedio (Varianza): %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Muestro la exactitud obtenida en cada fold de 10 Fold CV
np.set_printoptions(precision=4)
print("Exactitud de cada fold= {}".format(scores))

allData = np.load("allData_HOG.npy")
allClasses = np.load("allClasses_HOG.npy")
