import pandas as pd  # Importa la biblioteca pandas para manipulación de datos
from collections import Counter  # Importa Counter para contar elementos en una colección
import numpy as np  # Importa numpy para operaciones matemáticas avanzadas

print("PUNTO 1")  # Imprime un encabezado para identificar la sección del código

# Cargar datos desde el archivo CSV
url = 'https://drive.google.com/uc?export=download&id=1BQEFonHa5aYO4MTg1EWxGIuRSgCw6ZXb'  # URL del archivo CSV
datos = pd.read_csv(url, usecols=[0, 1, 2, 5, 12, 13], encoding='latin1')  # Carga las columnas seleccionadas del archivo CSV

# Filtrar por edad entre 40 y 45
datos_filtrados = datos[(datos['Edad'] >= 40) & (datos['Edad'] <= 45)].copy()  # Filtra filas donde la edad está entre 40 y 45

# Usar solo columnas categóricas
datos_filtrados = datos_filtrados.select_dtypes(include='object')  # Selecciona solo columnas categóricas
#print("Cantidad de datos sin eliminar:",len(datos_filtrados))
datos_filtrados = datos_filtrados.dropna()  # Elimina filas con valores faltantes
# print("Cantidad de datos con eliminados:",len(datos_filtrados))
# Mezclar datos aleatoriamente y dividir en entrenamiento y prueba (80/20)
datos_filtrados = datos_filtrados.sample(frac=1, random_state=42).reset_index(drop=True)  # Mezcla los datos aleatoriamente
split_index = int(0.8 * len(datos_filtrados))  # Calcula el índice para dividir los datos en 80/20
train_data = datos_filtrados.iloc[:split_index]  # Datos de entrenamiento (80%)
test_data = datos_filtrados.iloc[split_index:]  # Datos de prueba (20%)
#print("Cantidad de datos entrenamiento:",len(train_data))
#print("Cantidad de datos prueba:",len(test_data))
# Entropía de Shannon
def entropy(columna):
    print("columna \n",columna)
    conteos = Counter(columna)  # Cuenta la frecuencia de cada valor en la columna
    print("conteos",conteos)
    total = len(columna)  # Calcula el total de elementos en la columna
    return -sum((c / total) * np.log2(c / total) for c in conteos.values())  # Calcula la entropía de Shannon

# Ganancia de información
def info_gain(data, atributo, clase):
    total_ent = entropy(data[clase])  # Calcula la entropía total de la clase
    print("total_entropia   ",total_ent)
    valores = data[atributo].unique()  # Obtiene los valores únicos del atributo
    print("valores atributo sin unique    \n",data[atributo])
    print("valores con unique    ",valores)
    entropia_ponderada = 0  # Inicializa la entropía ponderada
    for val in valores:  # Itera sobre cada valor único del atributo
        subconjunto = data[data[atributo] == val]  # Filtra el subconjunto de datos con el valor actual
        peso = len(subconjunto) / len(data)  # Calcula el peso del subconjunto
        entropia_ponderada += peso * entropy(subconjunto[clase])  # Suma la entropía ponderada del subconjunto
    print("ganancia",f"{(total_ent - entropia_ponderada):.10f}" )    
    return total_ent - entropia_ponderada  # Retorna la ganancia de información

# Algoritmo ID3 recursivo
def id3(data, clase, atributos):
    if len(data[clase].unique()) == 1:  # Si todas las filas tienen la misma clase
        return data[clase].iloc[0]  # Retorna esa clase

    if not atributos:  # Si no quedan atributos para dividir
        return data[clase].mode()[0]  # Retorna la clase más frecuente

    ganancias = [info_gain(data, att, clase) for att in atributos]  # Calcula la ganancia de información para cada atributo
    mejor_atributo = atributos[np.argmax(ganancias)]  # Selecciona el atributo con mayor ganancia
    arbol = {mejor_atributo: {}}  # Crea un nodo del árbol con el mejor atributo

    for valor in data[mejor_atributo].unique():  # Itera sobre los valores únicos del mejor atributo
        subconjunto = data[data[mejor_atributo] == valor]  # Filtra el subconjunto de datos con el valor actual
        nuevos_atributos = [att for att in atributos if att != mejor_atributo]  # Excluye el mejor atributo de la lista
        arbol[mejor_atributo][valor] = id3(subconjunto, clase, nuevos_atributos)  # Llama recursivamente a ID3

    return arbol  # Retorna el árbol generado

# Función de predicción
def predict(arbol, fila):
    if not isinstance(arbol, dict):  # Si el árbol es una hoja
        return arbol  # Retorna el valor de la hoja
    atributo = next(iter(arbol))  # Obtiene el atributo del nodo actual
    valor = fila.get(atributo)  # Obtiene el valor del atributo en la fila
    if valor in arbol[atributo]:  # Si el valor está en el árbol
        return predict(arbol[atributo][valor], fila)  # Llama recursivamente a predict
    else:
        return None  # Retorna None si el valor no está en el árbol

# Entrenamiento
columna_clase = 'Estado'  # Define la columna de la clase objetivo
atributos = [col for col in train_data.columns if col != columna_clase]  # Lista de atributos excluyendo la clase
arbol = id3(train_data, columna_clase, atributos)  # Genera el árbol de decisión usando ID3

# Predicción sobre test_data
y_true = test_data[columna_clase].tolist()  # Lista de valores reales de la clase en los datos de prueba
y_pred = [predict(arbol, fila) for _, fila in test_data.iterrows()]  # Lista de predicciones para cada fila en los datos de prueba

# Prints de depuración
print("Distribución real en test_data (Estado):")  # Imprime la distribución de la clase en los datos de prueba
print(test_data['Estado'].value_counts())  # Cuenta los valores únicos de la clase en los datos de prueba

print("\nValores únicos predichos por el árbol:")  # Imprime los valores únicos predichos por el árbol
print(set(y_pred))  # Convierte las predicciones a un conjunto para obtener valores únicos

print("\nComparación real vs predicción (primeros 10 casos):")  # Imprime las primeras 10 comparaciones entre valores reales y predichos
for real, pred in list(zip(y_true, y_pred))[:10]:  # Itera sobre las primeras 10 predicciones
    print(f"Real: {real}  ->  Predicho: {pred}")  # Imprime el valor real y el predicho

# Métricas de evaluación
def confusion_metrics(y_true, y_pred, positivo='OTORGADO'):
    TP = sum(yt == yp == positivo for yt, yp in zip(y_true, y_pred))  # Verdaderos positivos
    TN = sum(yt == yp and yt != positivo for yt, yp in zip(y_true, y_pred))  # Verdaderos negativos
    FP = sum(yt != positivo and yp == positivo for yt, yp in zip(y_true, y_pred))  # Falsos positivos
    FN = sum(yt == positivo and yp != positivo for yt, yp in zip(y_true, y_pred))  # Falsos negativos

    accuracy = (TP + TN) / len(y_true)  # Precisión
    precision = TP / (TP + FP) if (TP + FP) else 0  # Precisión
    recall = TP / (TP + FN) if (TP + FN) else 0  # Sensibilidad
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0  # Puntaje F1

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,  # Retorna las métricas calculadas
            'accuracy': accuracy, 'f1_score': f1}

# Mostrar resultados
resultados = confusion_metrics(y_true, y_pred, positivo='OTORGADO')  # Calcula las métricas de evaluación
print("\nÁrbol generado:\n", arbol)  # Imprime el árbol generado
print("\nMatriz de Confusión y Métricas:")  # Imprime las métricas de evaluación
for clave, valor in resultados.items():  # Itera sobre las métricas
    print(f"{clave}: {valor:.4f}" if isinstance(valor, float) else f"{clave}: {valor}")  # Imprime cada métrica









    # Función para imprimir el árbol de forma jerárquica
from graphviz import Digraph

def visualizar_arbol(arbol, nombre_archivo="arbol"):
    dot = Digraph()
    contador = [0]  # Contador de nodos únicos

    def agregar_nodo(subarbol, padre=None, valor_padre=None):
        nodo_id = str(contador[0])
        contador[0] += 1

        if isinstance(subarbol, dict):
            atributo = next(iter(subarbol))
            dot.node(nodo_id, atributo)  # Nodo del atributo
            if padre is not None:
                dot.edge(padre, nodo_id, label=str(valor_padre))
            for valor, rama in subarbol[atributo].items():
                agregar_nodo(rama, nodo_id, valor)
        else:
            dot.node(nodo_id, str(subarbol), shape='box')  # Nodo hoja
            if padre is not None:
                dot.edge(padre, nodo_id, label=str(valor_padre))

    agregar_nodo(arbol)
    dot.render(filename=nombre_archivo, format='png', cleanup=True)
    print(f"Árbol guardado como {nombre_archivo}.png")

# Llamar a la función
visualizar_arbol(arbol)
