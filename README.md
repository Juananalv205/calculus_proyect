# Calculus_Proyect
# Regresión Lineal Múltiple

Este proyecto implementa una regresión lineal múltiple en Python utilizando la librería scikit-learn para el análisis de datos multivariable. El código se enfoca en predecir la variable dependiente (`TOTAL`) a partir de varias variables independientes (`COD_PROD_04`, `COD_PROD_05`, `COD_PROD_06`) Los cuales refieren a 3 categorías diferentes de productos. A continuación, se presenta una explicación detallada del código.

## 1. Importar Librerías

```python
import pandas as pd
import csv
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```
Se importan las librerías necesarias para el análisis de datos, manipulación de matrices, implementación de regresión lineal y visualización de resultados.

## 2. Obtener Datos de Origen

```python
df = pd.read_excel('dataset.xlsx', index_col=0)
```
Se carga el conjunto de datos desde el archivo Excel 'dataset.xlsx' y se imprime para verificar la correcta lectura.

## 3. Selección de Columnas de Trabajo
```python
y_multiple = df['TOTAL']
x_multiple = df[['COD_PROD_04', 'COD_PROD_05', 'COD_PROD_06']]
```
Se seleccionan las variables dependientes e independientes del conjunto de datos.

## 4. Definición del Algoritmo a Utilizar

```python
reg = LinearRegression()
```
Se instancia un objeto de la clase LinearRegression de scikit-learn para utilizar el algoritmo de regresión lineal.

## 5. Entrenamiento del Modelo
```python
reg = reg.fit(x_multiple, y_multiple)
```
Se entrena el modelo utilizando las variables independientes (x_multiple) y la variable dependiente (y_multiple).

## 6. Ejecución de Predicción

```python
y_pred = reg.predict(x_multiple)
```python
Se realiza una predicción utilizando las variables independientes.

## 7. Impresión de Información Resumen

```python
print('DATOS DEL MODELO REGRESIÓN LINEAL MÚLTIPLE')
print('Margen de Error del Modelo:')
print(np.sqrt(mean_squared_error(y_multiple, y_pred)))

print('Precisión del Modelo:')
print(reg.score(x_multiple, y_multiple))

print('Valor de la Pendiente o Coeficiente "a":')
print(reg.coef_)

print('Valor de la Intersección o Coeficiente "b":')
print(reg.intercept_)
```
Se imprime información relevante del modelo, como el margen de error, precisión, coeficientes de la regresión y la intersección.

## 8. Exportación a Archivo CSV

```python
myData = [['COD_PROD_04', 'COD_PROD_05', 'COD_PROD_06', 'Constante'], [reg.coef_, reg.intercept_]]
myFile = open('Ecuación Final.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)
```
Los coeficientes y la intersección del modelo se exportan a un archivo CSV llamado 'Regresion_Multiple01.csv'.

## 9. Predicción a Nivel de Prueba
```python
COD_PROD_04 = 1000
COD_PROD_05 = 100
COD_PROD_06 = 20
datos_entrada = pd.DataFrame({'COD_PROD_04': [COD_PROD_04], 'COD_PROD_05': [COD_PROD_05], 'COD_PROD_06': [COD_PROD_06]})
resultado_prediccion = reg.predict(datos_entrada)
print('RESULTADO DE LA PREDICCIÓN')
print("Valor : \n", resultado_prediccion)
print("-"*50)
```
Se realiza una predicción de prueba utilizando valores específicos para las variables independientes.
---
Este código proporciona una implementación básica de regresión lineal múltiple y puede ser utilizado como punto de partida para análisis más avanzados.
