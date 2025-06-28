import pandas as pd  # Librería para manipulación y análisis de datos
import numpy as np  # Librería para operaciones numéricas
from collections import Counter  # Herramienta para contar elementos en colecciones
import matplotlib.pyplot as plt  # Librería para visualización de datos

# pip install pandas numpy scikit-learn matplotlib graphviz

# --- Carga y Preprocesamiento de Datos ---
def cargar_datos(url, columnas, codificacion='latin1'):
    """
    Carga un CSV desde una URL y devuelve un DataFrame con las columnas especificadas.
    """  # Carga el archivo CSV con las columnas y codificación especificadas
    df = pd.read_csv(url, usecols=columnas, encoding=codificacion) 
         # Muestra información sobre los datos cargados
    print(f"[cargar_datos] Datos cargados con {len(df)} filas y {len(df.columns)} columnas.")  
    return df  # Devuelve el DataFrame cargado

def filtrar_edad(df, columna_edad, edad_min, edad_max):
    """
    Filtra el DataFrame para incluir solo filas cuya edad esté entre edad_min y edad_max.
    """             # Filtra las filas según el rango de edad
    filtrado = df[(df[columna_edad] >= edad_min) & (df[columna_edad] <= edad_max)].copy() 
                    # Muestra el número de filas filtradas
    print(f"[filtrar_edad] Filtrado: {len(filtrado)} filas entre {edad_min} y {edad_max} años.")  
    return filtrado  # Devuelve el DataFrame filtrado

def preprocesar_categoricas(df, columna_target):
    """
    Selecciona solo columnas categóricas y elimina filas sin valor en la variable target.
    """
    # Selecciona columnas categóricas y elimina filas con valores nulos en la columna objetivo
    categ = df.select_dtypes(include='object').dropna(subset=[columna_target]) 
    # Muestra las columnas categóricas y el número de filas restantes 
    print(f"[preprocesar_categoricas] Columnas categóricas: {list(categ.columns)}. Filas tras dropna: {len(categ)}.")  
    # Devuelve el DataFrame procesado
    return categ  
#################################################### EJERCICIO 1 PUNTO 1  ####################################################################

def dividir_entrenamiento_prueba(df, prueba_size=0.2, random_state=None):
    """
    Mezcla aleatoriamente el DataFrame y lo divide en entrenamiento y prueba según prueba_size.
    """
    # Mezcla aleatoriamente las filas del DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  
    # Calcula el índice para dividir el DataFrame
    idx = int(len(df_shuffled)*(1-prueba_size))  
    # Selecciona las filas para el conjunto de entrenamiento
    entrenamiento = df_shuffled.iloc[:idx]  
    # Selecciona las filas para el conjunto de prueba
    prueba  = df_shuffled.iloc[idx:]  
    # Muestra el tamaño de los conjuntos
    print(f"Entrenamiento:{len(entrenamiento)} filas, Prueba:{len(prueba)} filas.") 
    # Devuelve los conjuntos de entrenamiento y prueba
    return entrenamiento, prueba  

#################################################### EJERCICIO 1 PUNTO 2  ####################################################################
# --- Implementación ID3 ---
def entropia(serie):
    """
    Calcula la entropía de Shannon de una serie de etiquetas.
    medida cuantitativa de la cantidad de informaci´on que contiene una variable
    """
    # Cuenta la frecuencia de cada valor en la serie (columna de estado, ACEPTADO o RECHAZADO)
    conteos = Counter(serie)  
    total = len(serie)  # Calcula el número total de elementos
    # Calcula la entropía usando la fórmula de Shannon, la cantidad de veces que tenemos por cada estado
    resEntropia = -sum((cantEstado/total)*np.log2(cantEstado/total) for cantEstado in conteos.values())  
    # Muestra los valores y la entropía calculada
    print(f"[entropia] Valores: {dict(conteos)}, Entropía: {resEntropia:.4f}") 
    return resEntropia  # Devuelve la entropía calculada

def ganancia_informacion(df, atributo, target):
    """
    Calcula la ganancia de información de particionar df por caracteristica.
    """
     # Calcula la entropía total del conjunto de datos
    ent_total = entropia(df[target]) 
    # Calcula la entropía ponderada por partición
    ent_ponderada = sum((len(sub)/len(df))*entropia(sub[target]) for _, sub in df.groupby(atributo))  
    # Calcula la ganancia de información
    ganancia = ent_total - ent_ponderada  
    # Muestra la característica y su ganancia de información
    print(f"[ganancia_informacion] Feature: {atributo}, Ganancia: {ganancia:.4f}")  
    return ganancia  # Devuelve la ganancia de información

def construir_id3(df, target, atributos, profundidad_max=None, profundidad_actual=0):
    """
    Construye recursivamente un árbol de decisión usando ID3 con límite de profundidad.
    """
    # Caso base: nodo puro
    if len(df[target].unique()) == 1:
        clase = df[target].iloc[0]
        return clase

    # Caso base: sin atributos o se alcanzó la profundidad máxima
    if not atributos or (profundidad_max is not None and profundidad_actual >= profundidad_max):
        moda = df[target].mode()[0]
        return moda

    # Elegir mejor atributo según ganancia de información
    ganancias = {atributo: ganancia_informacion(df, atributo, target) for atributo in atributos}
    atributo_mejor_ganancia = max(ganancias, key=ganancias.get)
    arbol = {atributo_mejor_ganancia: {}}

    for valor, sub in df.groupby(atributo_mejor_ganancia):
        arbol[atributo_mejor_ganancia][valor] = construir_id3(
            sub,
            target,
            [a for a in atributos if a != atributo_mejor_ganancia],
            profundidad_max,
            profundidad_actual + 1
        )
    return arbol

def predecir_id3(arbol, fila, primer_valor_de_moda):
    """
    Predice la clase de una instancia usando el árbol ID3.
    """
    if not isinstance(arbol, dict):
        return arbol or primer_valor_de_moda
    caracteristica = next(iter(arbol))
    valor = fila.get(caracteristica)
    rama = arbol[caracteristica].get(valor)
    return predecir_id3(rama, fila, primer_valor_de_moda)


###### Matriz de Confusión Acurracy F1-score ### EJERCICIO 1 PUNTOS  3 y 4 - EJERCICIO 2 PUNTOS 2 Y 3###############
# --- Evaluación de Modelos ---

def evaluar(y_true, y_pred):
    """
    Calcula manualmente la matriz de confusión y métricas de evaluación.
    """
    # Asumimos dos clases distintas
    clases = sorted(set(y_true))
    if len(clases) != 2:
        raise ValueError("Esta función solo soporta clasificación binaria.")
    
    pos_label = clases[0]
    neg_label = clases[1]
    
    TP = sum((yt == pos_label and yp == pos_label) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == neg_label and yp == neg_label) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == neg_label and yp == pos_label) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == pos_label and yp == neg_label) for yt, yp in zip(y_true, y_pred))

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Mostrar matriz de confusión de forma tabular
    print(f"[evaluar] Matriz de Confusión:")
    print(f"              Predicho")
    print(f"              {pos_label}    {neg_label}")
    print(f"Real {pos_label}    {TP}        {FN}")
    print(f"Real {neg_label}    {FP}        {TN}")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'cm': [[TP, FN], [FP, TN]],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


#########Random Forest ####################### EJERCICIO 2 PUNTO 1 #################################################
# --- Random Forest ---
# Random Forest manual
def construir_bosque(df, target, atributos, n_arboles=10, profundidad_max=None):
    bosque = []
    for _ in range(n_arboles):
        muestra = df.sample(frac=1, replace=True)
        arbol = construir_id3(muestra, target, atributos, profundidad_max)
        bosque.append(arbol)
    return bosque

def predecir_bosque(bosque, df_test, clase_defecto):
    predicciones = []
    for _, fila in df_test.iterrows():
        votos = [predecir_id3(arbol, fila, clase_defecto) for arbol in bosque]
        predicciones.append(Counter(votos).most_common(1)[0][0])
    return predicciones


def contar_nodos(arbol):
    if not isinstance(arbol, dict):
        return 1
    nodos = 0
    for rama in arbol.values():
        for subarbol in rama.values():
            nodos += contar_nodos(subarbol)
    return nodos + 1  # sumar el nodo raíz


# Gráfico precisión vs profundidad
def graficar_precision_vs_tamano_arbol(df_train, df_test, target):
    atributos = [c for c in df_train.columns if c != target]
    clase_defecto = df_train[target].mode()[0]
    tamanos = []
    precisiones_train = []
    precisiones_test = []

    for _ in range(10):  # repetir para distintos árboles individuales
        muestra = df_train.sample(frac=1, replace=True)
        arbol = construir_id3(muestra, target, atributos)
        tamano = contar_nodos(arbol)
        pred_train = [predecir_id3(arbol, fila, clase_defecto) for _, fila in df_train.iterrows()]
        pred_test = [predecir_id3(arbol, fila, clase_defecto) for _, fila in df_test.iterrows()]
        prec_train = sum(yt == yp for yt, yp in zip(df_train[target], pred_train)) / len(df_train)
        prec_test = sum(yt == yp for yt, yp in zip(df_test[target], pred_test)) / len(df_test)

        tamanos.append(tamano)
        precisiones_train.append(prec_train)
        precisiones_test.append(prec_test)

    plt.plot(tamanos, precisiones_train, 'o-', label='Train')
    plt.plot(tamanos, precisiones_test, 'o-', label='Test')
    plt.title('Precisión vs Tamaño del Árbol')
    plt.xlabel('Tamaño del Árbol (número de nodos)')
    plt.ylabel('Precisión')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Ejecución Principal ---

URL = 'https://drive.google.com/uc?export=download&id=1BQEFonHa5aYO4MTg1EWxGIuRSgCw6ZXb'
COLS = [0,1,2,5,12,13]
TARGET = 'Estado'
# --- Pipeline completo ---
df = cargar_datos(URL, COLS)
df = filtrar_edad(df, 'Edad', 40, 45)
df = preprocesar_categoricas(df, TARGET)
entrenamiento, prueba = dividir_entrenamiento_prueba(df)
######## ID3 #############################EJERCICIO 1 PUNTO 2 ##########################################################

atributos = [atributo for atributo in entrenamiento.columns if atributo != TARGET]
primer_valor_de_moda = entrenamiento[TARGET].mode()[0]
arbol_id3 = construir_id3(entrenamiento, TARGET, atributos)
y_pred_id3 = [predecir_id3(arbol_id3, fila, primer_valor_de_moda) for _, fila in prueba.iterrows()]

###### Matriz de Confusión Acurracy F1-score ### EJERCICIO 1 PUNTO 3 y 4 #############################################
resultados_id3 = evaluar(prueba[TARGET].tolist(), y_pred_id3)


#########Random Forest ####################### EJERCICIO 2 PUNTO 1 #################################################

print("\n--- Random Forest Manual ---")
bosque = construir_bosque(entrenamiento, TARGET, atributos, n_arboles=10)
y_pred = predecir_bosque(bosque, prueba, primer_valor_de_moda)
evaluar(prueba[TARGET].tolist(), y_pred)

print("\n--- Gráfico de Precisión vs Profundidad ---")
graficar_precision_vs_tamano_arbol(entrenamiento, prueba, TARGET)

