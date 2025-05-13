import pandas as pd  # Librería para manipulación y análisis de datos
import numpy as np  # Librería para operaciones numéricas
from collections import Counter  # Herramienta para contar elementos en colecciones
from sklearn.ensemble import RandomForestClassifier  # Modelo de clasificación Random Forest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # Métricas de evaluación
import matplotlib.pyplot as plt  # Librería para visualización de datos
from graphviz import Digraph  # Herramienta para crear gráficos de árboles

# pip install pandas numpy scikit-learn matplotlib graphviz

# --- Carga y Preprocesamiento de Datos ---
def cargar_datos(url, columnas, codificacion='latin1'):
    """
    Carga un CSV desde una URL y devuelve un DataFrame con las columnas especificadas.
    """
    df = pd.read_csv(url, usecols=columnas, encoding=codificacion)  # Carga el archivo CSV con las columnas y codificación especificadas
    print(f"[cargar_datos] Datos cargados con {len(df)} filas y {len(df.columns)} columnas.")  # Muestra información sobre los datos cargados
    return df  # Devuelve el DataFrame cargado

def filtrar_edad(df, columna_edad, edad_min, edad_max):
    """
    Filtra el DataFrame para incluir solo filas cuya edad esté entre edad_min y edad_max.
    """
    filtrado = df[(df[columna_edad] >= edad_min) & (df[columna_edad] <= edad_max)].copy()  # Filtra las filas según el rango de edad
    print(f"[filtrar_edad] Filtrado: {len(filtrado)} filas entre {edad_min} y {edad_max} años.")  # Muestra el número de filas filtradas
    return filtrado  # Devuelve el DataFrame filtrado

def preprocesar_categoricas(df, columna_target):
    """
    Selecciona solo columnas categóricas y elimina filas sin valor en la variable target.
    """
    categ = df.select_dtypes(include='object').dropna(subset=[columna_target])  # Selecciona columnas categóricas y elimina filas con valores nulos en la columna objetivo
    print(f"[preprocesar_categoricas] Columnas categóricas: {list(categ.columns)}. Filas tras dropna: {len(categ)}.")  # Muestra las columnas categóricas y el número de filas restantes
    return categ  # Devuelve el DataFrame procesado
#################################################### EJERCICIO 1 PUNTO 1  ####################################################################

def dividir_entrenamiento_prueba(df, prueba_size=0.2, random_state=None):
    """
    Mezcla aleatoriamente el DataFrame y lo divide en entrenamiento y prueba según prueba_size.
    """
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Mezcla aleatoriamente las filas del DataFrame
    idx = int(len(df_shuffled)*(1-prueba_size))  # Calcula el índice para dividir el DataFrame
    entrenamiento = df_shuffled.iloc[:idx]  # Selecciona las filas para el conjunto de entrenamiento
    prueba  = df_shuffled.iloc[idx:]  # Selecciona las filas para el conjunto de prueba
    print(f"[dividir_entrenamiento_prueba] Entrenamiento:{len(entrenamiento)} filas, Prueba:{len(prueba)} filas.")  # Muestra el tamaño de los conjuntos
    return entrenamiento, prueba  # Devuelve los conjuntos de entrenamiento y prueba

#################################################### EJERCICIO 1 PUNTO 2  ####################################################################
# --- Implementación ID3 ---
def entropia(serie):
    """
    Calcula la entropía de Shannon de una serie de etiquetas.
    medida cuantitativa de la cantidad de informaci´on que contiene una variable
    """
    conteos = Counter(serie)  # Cuenta la frecuencia de cada valor en la serie (columna de estado, ACEPTADO o RECHAZADO)
    total = len(serie)  # Calcula el número total de elementos
    resultadoEntropia = -sum((cantEstado/total)*np.log2(cantEstado/total) for cantEstado in conteos.values())  # Calcula la entropía usando la fórmula de Shannon, la cantidad de veces que tenemos por cada estado
    print(f"[entropia] Valores: {dict(conteos)}, Entropía: {resultadoEntropia:.4f}")  # Muestra los valores y la entropía calculada
    return resultadoEntropia  # Devuelve la entropía calculada

def ganancia_informacion(df, atributo, target):
    """
    Calcula la ganancia de información de particionar df por caracteristica.
    """
    ent_total = entropia(df[target])  # Calcula la entropía total del conjunto de datos
    ent_ponderada = sum((len(sub)/len(df))*entropia(sub[target]) for _, sub in df.groupby(atributo))  # Calcula la entropía ponderada por partición
    ganancia = ent_total - ent_ponderada  # Calcula la ganancia de información
    print(f"[ganancia_informacion] Feature: {atributo}, Ganancia: {ganancia:.4f}")  # Muestra la característica y su ganancia de información
    return ganancia  # Devuelve la ganancia de información

def construir_id3(df, target, atributos):  # caracteristicas son los atributos 
    """
    Construye recursivamente un árbol de decisión usando ID3.
    """
    # Caso base: si todas las etiquetas iguales
    if len(df[target].unique()) == 1:    # unique unifica para eliminar duplicados y si hay un solo elemento es una hoja o nodo puro
        clase = df[target].iloc[0]
        print(f"[construir_id3] Nodo hoja con clase {clase}.")
        return clase
    # Sin atributos remanentes
    if not atributos:           # no tiene valor el atributo en esa fila
        moda = df[target].mode()[0]   # lo etiqueta con el más frecuente
        print(f"[construir_id3] Sin atributos, retorna moda {moda}.")
        return moda

    # Elegir mejor atributo de acuerdo a la ganancia de información de cada uno
    #Se elige el atributo con mayor valor para la ganancia de informaci´on.
    ganancias = {atributo: ganancia_informacion(df, atributo, target) for atributo in atributos}
    atributo_mejor_ganancia = max(ganancias, key=ganancias.get)
    print(f"[construir_id3] Mejor atributos: {atributo_mejor_ganancia}.")
    arbol = {atributo_mejor_ganancia: {}} 

    # Particionar 
    #Se calcula la entrop´ıa y ganancia de los  nodos hijos 

    for valor, sub in df.groupby(atributo_mejor_ganancia):
        # se utiliza la recursividad para armar el árbol
                                               # construir_id3 (df, target, atributos)  
        arbol[atributo_mejor_ganancia][valor] = construir_id3(  
            sub,
            target,
            [f for f in atributos if f != atributo_mejor_ganancia]
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

# def evaluar(y_true, y_pred):
#     """
#     Calcula matriz de confusión y métricas (accuracy, precision, recall, F1).
#     """
#     etiquetas = sorted(set(y_true))
#     cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, pos_label=etiquetas[0])
#     rec = recall_score(y_true, y_pred, pos_label=etiquetas[0])
#     f1 = f1_score(y_true, y_pred, pos_label=etiquetas[0])
#     print(f"[evaluar] CM:\n _Matriz de Confusión {cm}\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#     return {'cm': cm, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


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
def ejecutar_bosque_random(X_train, y_train, X_test, n_estimators=10, random_state=None):
    """
    Entrena y predice con RandomForestClassifier.
    """
    modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    print(f"[ejecutar_bosque_random] Predicciones generadas para {len(preds)} instancias.")
    return preds



#####curva_precision#################### EJERCICIO 2 PUNTO 4 ############################################################# 
def graficar_curva_precision(X_train, y_train, X_test, y_test, max_arboles=10):
    """
    Grafica precisión vs número de árboles para train y test.
    """
    arboles = list(range(1, max_arboles+1))
    p_train, p_test = [], []
    etiqueta = y_train.mode()[0]
    for n in arboles:
        rf = RandomForestClassifier(n_estimators=n, random_state=None)
        rf.fit(X_train, y_train)
        p_train.append(precision_score(y_train, rf.predict(X_train), pos_label=etiqueta))
        p_test.append(precision_score(y_test, rf.predict(X_test), pos_label=etiqueta))
    plt.plot(arboles, p_train, marker='o', label='Entrenamiento')
    plt.plot(arboles, p_test, marker='o', label='Prueba')
    plt.xlabel('Número de árboles')
    plt.ylabel('Precisión')
    plt.title('Precisión vs tamaño del bosque')
    plt.legend()
    plt.show()

# --- Visualización de Árbol ID3 ---
def visualizar_arbol(arbol, nombre_archivo='arbol'):
    """
    Guarda una representación gráfica del árbol ID3 usando graphviz.
    """
    dot = Digraph()
    idx = {'i': 0}
    def agregar_nodos(sub, padre=None, etiqueta=None):
        nodo = str(idx['i']); idx['i'] += 1
        if isinstance(sub, dict):
            feat = next(iter(sub))
            dot.node(nodo, feat)
            if padre: dot.edge(padre, nodo, label=str(etiqueta))
            for val, ch in sub[feat].items():
                agregar_nodos(ch, nodo, val)
        else:
            dot.node(nodo, str(sub), shape='box')
            if padre: dot.edge(padre, nodo, label=str(etiqueta))
    agregar_nodos(arbol)
    dot.render(filename=nombre_archivo, format='png', cleanup=True)
    print(f"[visualizar_arbol] Árbol guardado en {nombre_archivo}.png")

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
# visualizar_arbol(arbol_id3)
print(f"\n--- Predicción Random Forest ---")
X_tr = pd.get_dummies(entrenamiento.drop(columns=[TARGET]))
X_te = pd.get_dummies(prueba.drop(columns=[TARGET]))
X_train, X_test = X_tr.align(X_te, join='left', axis=1, fill_value=0)
y_pred_rf = ejecutar_bosque_random(X_train, entrenamiento[TARGET], X_test)

###### Matriz de Confusión Acurracy F1-score ##### EJERCICIO 2 PUNTO 2 y 3 #############################################
resultados_rf = evaluar(prueba[TARGET].tolist(), y_pred_rf)

######curva_precision############## EJERCICIO 2 PUNTO 4 ############################################################# 
graficar_curva_precision(X_train, entrenamiento[TARGET], X_test, prueba[TARGET])
