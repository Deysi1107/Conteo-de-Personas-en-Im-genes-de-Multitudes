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

# Seteo de paths y archivos de prueba
PATH_POSITIVE_TRAIN="Base de datos/Train/Positivos/"
PATH_NEGATIVE_TRAIN="Base de datos/Train/Negativos/"

#Funciòn para la lectra de datos del entrenamiento

def loadTrainingData():
    """
    Retorna:
    trainingData -- matriz con las instancias
    classes --      vector con las clases de cada instancia
    """

    # Matriz de descriptores que deben ser completadas
    trainingData=np.array([])
    trainingDataNeg=np.array([])

    ################################################
    # Casos positivos
    # Obtengo la lista de casos positivos
    listFiles=[abspath(arch.path) for arch in scandir(PATH_POSITIVE_TRAIN) if arch.is_file()]
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
        # Agrego a trainingData
        trainingData=np.hstack((trainingData, h2))

    print("Leidas " + str(len(listFiles)) + " imágenes de entrenamiento -> positivas")
    # Hago un reshape
    trainingData=trainingData.reshape((len(listFiles), len(h2)))
    # Genero el vector de clases
    classes=np.ones(len(listFiles))

    ################################################
    # Casos negativos
    # Obtengo la lista de casos positivos
    listFilesNeg=[abspath(arch.path) for arch in scandir(PATH_NEGATIVE_TRAIN) if arch.is_file()]
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
        trainingDataNeg=np.hstack((trainingDataNeg, h2))

    print("Leidas " + str(len(listFilesNeg)) + " imágenes de entrenamiento -> negativas")
    # Hago un reshape
    trainingDataNeg=trainingDataNeg.reshape((len(listFilesNeg), len(h2)))
    # Genero el vector de clases
    classesNeg=np.zeros(len(listFilesNeg))

    # Merge de los datos
    # Matriz de features
    trainingData=np.concatenate((trainingData, trainingDataNeg), axis=0)
    # Vector de clases
    classes=np.concatenate((classes, classesNeg), axis=0)
    return trainingData, classes

# -----------------------------
# Ejecución de la prueba
# -----------------------------

# Obtengo los datos de trainning
X_Train, y_train = loadTrainingData()
X_Train.size
# Dimensiones :p
X_Train.ndim
# Alto por ancho
X_Train.shape

# Guardo los vectores obtenidos
np.save("X_Train_HOG", X_Train)
np.save("y_train_HOG", y_train)

# Creo una SVM con kernel linear
clf = SVC(kernel="linear", probability = True)

# Entreno la SVM
clf.fit(X_Train, y_train)

# Guardo el modelo entreando
dump(clf, 'HOG_clf.joblib')