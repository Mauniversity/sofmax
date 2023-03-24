# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:59:26 2023

@author: Mauro
"""

import matplotlib.pyplot as graf
import numpy as np
import random

import pandas as pd


# datos random
np.random.seed(42)

#funcion escalon
# def escalon(t):
#     if t >= 0:
#         return 1
#     return 0

# def softmax(z):
#     return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

# def prediccion(X, W, b):
#     return softmax(np.matmul(X, W) + b)

# TODO: Algoritmo Perceptron.
def perceptron_escalon(datos, W, b, alfa):
    
    for i in range(0,3000):
        w1=W[0]
        w2=W[1]
        k=random.randint(0,len(datos[:,0])-1)    
    
        
        #
        if w1*datos[k,0]+w2*datos[k,1]+b>0:        
    
            t=0
            
            if t!=datos[k,2]:
                w1=w1-alfa*datos[k,0]
                w2=w2-alfa*datos[k,1]
                b=b-alfa*1
            
        else:
            
            t=1
            
            if t!=datos[k,2]:
                w1=w1+alfa*datos[k,0]
                w2=w2+alfa*datos[k,1]
                b=b+alfa*1

            #  print("mal 1"+str(w))   
        W=[w1,w2]
    return W, b

#Lo siguiente es la parte del entrenamiento utilizando el anterior algoritmo del percptron.

def entrenamiento(X, y, learn_rate=0.001, num_epochs=20):
    """
    Esta funcion entrena el perceptron repetidamente en el dataset y retorna las lineas de clasificacion
    obtenidas en las iteraciones,
    
    """
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    
    # Lineas que seran ploteadas.
    lineas = []
    for i in range(num_epochs):
        # En cada epoch, se aplica el perceptron.
        W, b = perceptron_escalon(datos, W, b, learn_rate)
        lineas.append((-W[0]/W[1], -b/W[1]))
    return lineas, W, b


# Se inicia con la subida de datos desde el computador.

datos = np.asarray(pd.read_csv("data.csv", header=None))

X = datos[:,0:2]
y = datos[:,2]

# Se activa el entrenamiento:

resul = entrenamiento(X, y)
lineas2, W, b = resul

# se asignan algunos valores para futuros usos:

penybi=resul[0]
pesos=resul[1]

m= penybi[-1][0] 
b=penybi[-1][1] 


# parte donde se grafica los puntos y la respectiva recta  despues del entrenanimiento.



        
fx = np.arange(0, 1, 0.1)
graf.plot(fx, m*fx+b)

# a contiuacion se aplica el sofmax por cada punto obtenido de los datos.

matrizeul=[]
for j in range(0,len(datos[:,0])): 
    
    eul=np.exp(pesos[0]*datos[j,0]+pesos[1]*datos[j,1]+b)
    matrizeul.append([eul])
    
total=np.sum(matrizeul)

# aplico sofmax

sfmax=[]
for k in range(0,len(datos[:,0])):
    opsof=(np.exp(pesos[0]*datos[k,0]+pesos[1]*datos[k,1]+b))/total
    sfmax.append([opsof])
    
# se grafica las probabilidades:


x111=np.linspace(0,1,5)                            
y111=np.linspace(0,1,5)
#y111=np.linspace(0,0.016,10)
A,B=np.meshgrid(x111,y111)  
graf.xlim() 
graf.ylim()                           

z=np.array((W[0]*A)+(W[1]*B)+b)                   


graf.figure(figsize=(9,6))
graf.legend()
graf.contourf(A, B, z)

for i in range(0,len(datos[:,0])):
    
    if datos[i,2]==0:
        
        graf.plot(datos[i,0],datos[i,1],"o",color="red")
        
    else:
            
        graf.plot(datos[i,0],datos[i,1],"o",color="blue")
        
fx = np.arange(0, 1, 0.1)
graf.plot(fx, m*fx+b,linewidth=5, color="black")

    
    


