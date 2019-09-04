#!/usr/bin/env python
# coding: utf-8

import argparse  #import para pasar parametros por linea de comandos

import numpy as np
import pandas as pd
import sklearn
import math
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix

def entrenar_total(train_data, test_data, l2, ratio_rbf, eta, classification, outputs):
    train_mses = np.empty(5)
    train_ccrs = np.empty(5)
    test_mses = np.empty(5)
    test_ccrs = np.empty(5)

    for s in range(100,600,100):   
        print("-----------")
        print("Semilla: %d" % s)
        print("-----------")     
        np.random.seed(s)
        train_mses[s//100-1], test_mses[s//100-1], train_ccrs[s//100-1], test_ccrs[s//100-1], matriz_confusion =         entrenar(train_data, test_data, classification, ratio_rbf, l2, eta, outputs)
        print("MSE de entrenamiento: %f" % train_mses[s//100-1])
        print("MSE de test: %f" % test_mses[s//100-1])
        if classification:
            print("CCR de entrenamiento: %.2f%%" % train_ccrs[s//100-1])
            print("CCR de test: %.2f%%" % test_ccrs[s//100-1])

    print("\n*********************")
    print("Resumen de resultados")
    print("*********************")
    print("MSE de entrenamiento: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
    print("MSE de test: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))

    plt.title("MSE de train/test")
    plt.xlabel("Iteración")
    plt.ylabel("Error")
    plt.legend(('train', 'test'))
    plt.plot(np.arange(1, train_mses.shape[0]+1), train_mses, 'r')
    plt.plot(np.arange(1, test_mses.shape[0]+1), test_mses, 'b')
    plt.show()
        
    if classification:
        plt.xlabel("Iteración")
        plt.ylabel("Patrones")
        plt.title("CCR de train/test")
        plt.legend(('train', 'test'))
        plt.plot(np.arange(1, train_mses.shape[0]+1),train_ccrs, 'b')
        plt.plot(np.arange(1, test_mses.shape[0]+1),test_ccrs, 'r')
        plt.show()
        
        print("CCR de entrenamiento: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("CCR de test: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))
        print("La matriz de confusión es la siguiente:")
    
        plt.figure(figsize = (4,4))
        sn.heatmap(matriz_confusion, annot=True)
        plt.show()
    
    return 0;



def inicializarDatos(train_data, test_data, outputs):
    train_inputs = train_data.values[:, 0:-outputs];
    train_outputs = train_data.values[:, -outputs];#Guarda las clases existentes

    test_inputs = test_data.values[:, 0:-outputs];
    test_outputs = test_data.values[:, -outputs];
    
    return train_inputs, train_outputs, test_inputs, test_outputs

def inicializar_clas(train_inputs, train_outputs, num_rbfs):
    sss = StratifiedShuffleSplit(n_splits=num_rbfs, test_size=None, train_size=num_rbfs, random_state=0)

    for train_index, test_index in sss.split(train_inputs, train_outputs):
        centroides = train_inputs[train_index,:]
    
    indice = 0
   
    while centroides.shape[0] < num_rbfs:        
        centroides = np.r_[centroides, [train_inputs[test_index[indice]]]] 
        indice += 1
        
    while centroides.shape[0] > num_rbfs:
        centroides = centroides[np.random.choice(centroides.shape[0], num_rbfs,0),:]
    return centroides

def clustering(classification, train_inputs, train_outputs, num_rbfs):
    if classification == True:
        centros = inicializar_clas(train_inputs, train_outputs, num_rbfs)
    else:
        centros = train_inputs[np.random.choice(train_inputs.shape[0], num_rbfs, replace=False), :]
    
    kmedias = KMeans(n_init=1, max_iter=500, n_clusters=num_rbfs, init=centros).fit(train_inputs)
    centros = kmedias.cluster_centers_
    distancias = kmedias.transform(train_inputs)
    
    return kmedias, distancias, centros

def calcular_radios(centroides, num_rbfs):
    distancias_centros = squareform(pdist(centroides, 'euclidean'))
    radios = distancias_centros.sum(axis=1)
    radios = radios/(2*num_rbfs-1)
    
    return radios

def calcular_matriz_r(distancias, radios):
    matriz_r = np.ones(shape = (distancias.shape[0], distancias.shape[1]+1))
    matriz_r[:,:-1] =  np.exp(-(distancias**2)/(2*(np.power(radios, 2))))

    return matriz_r

def invertir_matriz_r(matriz_r, train_outputs):
    
    coefficients = np.transpose(np.dot(np.linalg.pinv(matriz_r), train_outputs))
    
    return coefficients

def calculate_logreg(matriz_r_train, train_outputs, eta, l2):
    if l2 == True:
        logreg = LogisticRegression(penalty='l2', C=1/eta, solver='liblinear', fit_intercept=False).fit(matriz_r_train,train_outputs)
    else:
        logreg = LogisticRegression(penalty='l1', C=1/eta, solver='liblinear', fit_intercept=False).fit(matriz_r_train,train_outputs)
    
    return logreg

def entrenar(train_data, test_data, classification, ratio_rbf, l2, eta, outputs):
    #Capa de entrada
    train_inputs, train_outputs, test_inputs, test_outputs = inicializarDatos(train_data, test_data, outputs)

    num_rbfs=int(round(ratio_rbf*len(train_inputs)))  #Multiplicar el ratio por el numero de patrones de train
    print("Número de RBFs utilizadas: %d" %(num_rbfs))
    print("---------------------------")

    kmedias, distancias, centros = clustering(classification, train_inputs, train_outputs, num_rbfs)
    radios = calcular_radios(centros, num_rbfs)
    matriz_r = calcular_matriz_r(distancias, radios)

    if not classification:
        coefficients = invertir_matriz_r(matriz_r, train_outputs)
    else:
        logreg = calculate_logreg(matriz_r, train_outputs, eta, l2)

    distancias_test = kmedias.transform(test_inputs)
    matriz_r_test = calcular_matriz_r(distancias_test, radios)
    
    if not classification:
        predicted_test_aux = np.dot(matriz_r_test, coefficients)
        predicted_train_aux = np.dot(matriz_r, coefficients)
        
        predicted_test= np.round(predicted_test_aux)
        predicted_train= np.round(predicted_train_aux) 

        predicted_test += 0.
        predicted_train += 0.
        
        test_mse = mean_squared_error(test_outputs, predicted_test_aux)
        train_mse = mean_squared_error(train_outputs, predicted_train_aux)
        
        train_ccr = 0
        test_ccr = 0
        matriz_confusion = np.ones(predicted_train.shape)
    else:
        predicted_train = logreg.predict(matriz_r)
        predicted_test = logreg.predict(matriz_r_test)
        
        train_mse = mean_squared_error(predicted_train, train_outputs)
        test_mse = mean_squared_error(predicted_test, test_outputs)
        
        train_ccr = logreg.score(matriz_r, train_outputs)*100
        test_ccr = logreg.score(matriz_r_test, test_outputs)*100
        
        #Calculamos la matriz de confusion
        matriz_confusion = confusion_matrix(test_outputs, predicted_test)
        
    return train_mse, test_mse, train_ccr, test_ccr, matriz_confusion

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Red Neuronal RBF. Número de iteraciones: 5")
    parser.add_argument("-t", "--train_file", help="File train data. [required]")
    parser.add_argument("-T", "--test_file", help="File test data. [required]")
    parser.add_argument("-c", "--classification", action="store_true", default=False, dest="classification", help="Classification option. [default:False]")
    parser.add_argument("-r", "--ratio_rbf", action="store", default=0.1, dest="ratio_rbf", help="Ratio of number of RBF for the model. [default:0.1]")
    parser.add_argument("-l", "--l2", action="store_true", default=False, dest="l2", help="Activate logistic regression type. [default:False]")
    parser.add_argument("-e", "--eta", action="store", default=0.01, dest="eta", help="Value of learning rate. [default:0.01]")
    parser.add_argument("-o", "--outputs", action="store", default=1, dest="outputs", help="Number of outputs. [default:1]")
    
    args=parser.parse_args()

    train_file = args.train_file
    train_data = pd.read_csv(train_file)
    if args.test_file:
        test_file = args.test_file
    else:
        print("Using train file for the test")
        test_file = args.train_file
    
    test_data = pd.read_csv(test_file)
    l2 = args.l2
    ratio_rbf = args.ratio_rbf
    eta = args.eta
    classification =args.classification
    outputs = args.outputs
    entrenar_total(train_data, test_data, l2, ratio_rbf, eta, classification, outputs)