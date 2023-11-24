#Regresión lineal múltiple

#1. Importar de librerias a utilizar
import pandas as pd
import csv
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#2. Obtener data de origen
df = pd.read_excel('Regresión Lineal Múltiple\dataset.xlsx', index_col=0)
print(df)
print(df.shape)
data_top=df.head()
df.head()

#3. Selección de columnas de trabajo
y_multiple = df['TOTAL']
x_multiple = df[['COD_PROD_04', 'COD_PROD_05', 'COD_PROD_06']]

#4. Definición del algoritmo a utilizar
reg = LinearRegression()

#5. Entrenamiento del modelo
reg = reg.fit(x_multiple, y_multiple)

#6. Ejecución de predicción
y_pred = reg.predict(x_multiple)

#7. Impresión de información resumen
print("-"*50)
print('DATOS DEL MODELO VECTORES DE SOPORTE REGRESIÓN')
print("-"*50)
print('Margen Error del modelo:')
print(np.sqrt(mean_squared_error(y_multiple, y_pred)))
print("-"*50)
print('Precisión del modelo:')
print(reg.score(x_multiple, y_multiple))
print("-"*50)
print('Valor de la pendiente o coeficiente "a":')
print(reg.coef_)
print("-"*50)
print('Valor de la intersección o coeficiente "b":')
print(reg.intercept_)
print("-"*50)

#8. Exportación a archivo csv
myData = [['COD_PROD_04', 'COD_PROD_05', 'COD_PROD_06', 'Constante'], [reg.coef_, reg.intercept_]]
myFile = open('Ecuación Final.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)

#9. Predicción a nivel de prueba
COD_PROD_04 = 1000
COD_PROD_05 = 100
COD_PROD_06 = 20
datos_entrada = pd.DataFrame({'COD_PROD_04': [COD_PROD_04], 'COD_PROD_05': [COD_PROD_05], 'COD_PROD_06': [COD_PROD_06]})
resultado_prediccion = reg.predict(datos_entrada)
print('RESULTADO DE LA PREDICCIÓN')
print("Valor Total: \n", resultado_prediccion)
print("-"*50)
