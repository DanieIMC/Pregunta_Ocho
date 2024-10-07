import pandas as pd
import numpy as np
from collections import Counter
#DATOS DE MYDATASET
# Altura (cm)
# Peso (kg)
# Talla (cm)
# Tamanio                  *****CLASE*****

# Cargar el dataset
df = pd.read_csv('C:/Users/s2dan/OneDrive/Documentos/WorkSpace/PrimerParcia_IA/MyDataset.csv')

# Función para calcular la entropía
def entropy(column):
    counts = Counter(column)
    total = len(column)
    ent = 0
    for count in counts.values():
        prob = count / total
        ent -= prob * np.log2(prob)
    return ent

# Función para calcular la ganancia de información
def information_gain(df, target_column, attribute_column):
    # Entropía del conjunto de datos completo
    total_entropy = entropy(df[target_column])
    
    # Entropía ponderada de los subconjuntos
    values = df[attribute_column].unique()
    weighted_entropy = 0
    for value in values:
        subset = df[df[attribute_column] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target_column])
    
    # Ganancia de información
    return total_entropy - weighted_entropy

# Cálculo de la entropía del dataset en la clase 'Tamanio'
total_entropy = entropy(df['Tamanio'])
print(f"Entropía total del conjunto de datos: {total_entropy}")

# Cálculo de la ganancia de información para cada atributo
attributes = ['Altura (cm)', 'Peso (kg)', 'Talla (cm)']
for attr in attributes:
    gain = information_gain(df, 'Tamanio', attr)
    print(f"Ganancia de información para {attr}: {gain}")

# Entropía: La función entropy calcula la entropía de una columna basada en las frecuencias de los valores únicos.

# Ganancia de información: La función information_gain primero calcula la entropía total del conjunto de datos,
# luego calcula la entropía ponderada de los subconjuntos definidos por el atributo en cuestión,
# y finalmente resta esta entropía ponderada de la entropía total.

