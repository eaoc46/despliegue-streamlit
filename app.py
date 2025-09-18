# -*- coding: utf-8 -*-
"""
Despliegue

- Cargamos el modelo
- Cargamos los datos futuros
- Preparar los datos futuros
- Aplicamos el modelo para la predicción
"""

#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cargamos el modelo
import pickle
filename = 'modelo-reg.pkl'
model_Tree, model_rf,model_Knn, model_NN, min_max_scaler, variables = pickle.load(open(filename, 'rb'))

#Cargamos los datos futuros
#data = pd.read_csv("videojuegos-datosFuturos.csv")
#data.head()

#Interfaz gráfica
#Se crea interfaz gráfica con streamlit para captura de los datos

import streamlit as st

st.title('Predicción de inversión en una tienda de videojuegos')

Edad = st.slider('Edad', min_value=14, max_value=52, value=20, step=1)
videojuego = st.selectbox('Videojuego', ["'Mass Effect'","'Battlefield'", "'Fifa'","'KOA: Reckoning'","'Crysis'","'Sim City'","'Dead Space'","'F1'"])
Plataforma = st.selectbox('Plataforma', ["'Play Station'", "'Xbox'","PC","Otros"])
Sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
Consumidor_habitual = st.selectbox('Consumidor_habitual', ['True', 'False'])


#Dataframe
datos = [[Edad, videojuego,Plataforma,Sexo,Consumidor_habitual]]
data = pd.DataFrame(datos, columns=['Edad', 'videojuego','Plataforma','Sexo','Consumidor_habitual']) #Dataframe con los mismos nombres de variables

#Se realiza la preparación
data_preparada=data.copy()

#En despliegue drop_first= False
data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma','Sexo', 'Consumidor_habitual'], drop_first=False, dtype=int)
data_preparada.head()

#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
data_preparada.head()

"""# **Predicciones**"""
#Hacemos la predicción con el Tree
Y_pred_TREE = model_Tree.predict(data_preparada)
Y_pred_RF = model_rf.predict(data_preparada)

print(Y_pred_TREE)
print(Y_pred_RF)

#Se normaliza la edad para predecir con Knn, Red
data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
data_preparada.head()

Y_pred_KNN = model_Knn.predict(data_preparada)
Y_pred_NN = model_NN.predict(data_preparada)

print(Y_pred_KNN)
print(Y_pred_NN)

data['Prediccion_TREE']=Y_pred_TREE
data['Prediccion_RF]=Y_pred_RF
data['Prediccion_KNN']=Y_pred_KNN
data['Prediccion_NN=Y_pred_NN

data.head()

#Predicciones finales
data
