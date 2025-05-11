import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from graphviz import Digraph

# --- Carga y Preprocesamiento de Datos ---
def cargar_datos(url, columnas, codificacion='latin1'):
    """
    Carga un CSV desde una URL y devuelve un DataFrame con las columnas especificadas.

    Teoría:
      - Preprocesamiento: paso inicial para asegurar formato y columnas correctas.
      - Codificación: manejo de caracteres especiales antes de análisis.
    """
    df = pd.read_csv(url, usecols=columnas, encoding=codificacion)
    print(f"[cargar_datos] Datos cargados con {len(df)} filas y {len(df.columns)} columnas.")
    return df


def filtrar_edad(df, columna_edad, edad_min, edad_max):
    """
    Filtra el DataFrame para incluir solo filas cuya edad esté entre edad_min y edad_max.

    Teoría:
      - Subconjunto relevante: enfocarse en el rango etario deseado mejora la especificidad.
    """
    filtrado = df[(df[columna_edad] >= edad_min) & (df[columna_edad] <= edad_max)].copy()
    print(f"[filtrar_edad] Filtrado: {len(filtrado)} filas entre {edad_min} y {edad_max} años.")
    return filtrado


def preprocesar_categoricas(df, columna_target):
    """
    Selecciona solo columnas categóricas y elimina filas sin valor en la variable target.

    Teoría:
      - Modelos ID3/RF con datos categóricos: conviene convertir variables a categorías.
      - Eliminación de nulos: necesaria para cálculos de entropía y métricas.
    """
    categ = df.select_dtypes(include='object').dropna(subset=[columna_target])
    print(f"[preprocesar_categoricas] Columnas categóricas: {list(categ.columns)}. Filas tras dropna: {len(categ)}.")
    return categ


def dividir_entrenamiento_prueba(df, prueba_size=0.2, random_state=None):
    """
    Mezcla aleatoriamente el DataFrame y lo divide en entrenamiento y prueba según prueba_size.

    Teoría:
      - Validación: split 80/20 para evaluar generalización del modelo.
      - Reproducibilidad: seed fija en sample.
    """
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    idx = int(len(df_shuffled)*(1-prueba_size))
    entrenamiento = df_shuffled.iloc[:idx]
    prueba  = df_shuffled.iloc[idx:]
    print(f"[dividir_entrenamiento_prueba] Entrenamiento:{len(entrenamiento)} filas, Prueba:{len(prueba)} filas.")
    return entrenamiento, prueba

# --- Implementación ID3 ---

def entropia(serie):
    """
    Calcula la entropía de Shannon de una serie de etiquetas.

    Teoría:
      H(S) = -∑ p_i log2(p_i), mide incertidumbre o impureza de S.
    """
    conteos = Counter(serie)
    total = len(serie)
    ent = -sum((c/total)*np.log2(c/total) for c in conteos.values())
    print(f"[entropia] Valores: {dict(conteos)}, Entropía: {ent:.4f}")
    return ent


def ganancia_informacion(df, caracteristica, target):
    """
    Calcula la ganancia de información de particionar df por caracteristica.

    Teoría:
      Gain(S,A) = H(S) - ∑ (|S_v|/|S|)*H(S_v), selecciona atributo que maximiza reducción de entropía.
    """
    ent_total = entropia(df[target])
    ent_ponderada = sum((len(sub)/len(df))*entropia(sub[target])
                         for _, sub in df.groupby(caracteristica))
    ganancia = ent_total - ent_ponderada
    print(f"[ganancia_informacion] Feature: {caracteristica}, Ganancia: {ganancia:.4f}")
    return ganancia


def construir_id3(df, target, caracteristicas):
    """
    Construye recursivamente un árbol de decisión usando ID3.

    Teoría:
      - Caso puro: nodo hoja.
      - Sin atributos: nodo con clase mayoritaria.
      - Seleccionar atributo con mayor ganancia de información.
      - Recursión sobre particiones.
    """
    # Caso base: si todas las etiquetas iguales
    if len(df[target].unique()) == 1:
        clase = df[target].iloc[0]
        print(f"[construir_id3] Nodo hoja con clase {clase}.")
        return clase
    # Sin atributos remanentes
    if not caracteristicas:
        moda = df[target].mode()[0]
        print(f"[construir_id3] Sin caracteristicas, retorna moda {moda}.")
        return moda

    # Elegir mejor atributo
    ganancias = {f: ganancia_informacion(df, f, target) for f in caracteristicas}
    mejor = max(ganancias, key=ganancias.get)
    print(f"[construir_id3] Mejor caracteristica: {mejor}.")
    arbol = {mejor: {}}

    # Particionar y recursar
    for valor, sub in df.groupby(mejor):
        arbol[mejor][valor] = construir_id3(
            sub,
            target,
            [f for f in caracteristicas if f != mejor]
        )
    return arbol


def predecir_id3(arbol, fila, por_defecto):
    """
    Predice la clase de una instancia usando el árbol ID3.

    Teoría:
      - Se recorre el árbol según el valor de cada atributo.
      - Si falta rama, retorna clase por defecto (mayoría).
    """
    if not isinstance(arbol, dict):
        return arbol or por_defecto
    caracteristica = next(iter(arbol))
    valor = fila.get(caracteristica)
    rama = arbol[caracteristica].get(valor)
    return predecir_id3(rama, fila, por_defecto)

# --- Evaluación de Modelos ---

def evaluar(y_true, y_pred):
    """
    Calcula matriz de confusión y métricas (accuracy, precision, recall, F1).

    Teoría:
      - Matriz de confusión: TP, FP, TN, FN.
      - Accuracy = (TP+TN)/total
      - Precision = TP/(TP+FP)
      - Recall = TP/(TP+FN)
      - F1 = 2*(precision*recall)/(precision+recall)
    """
    etiquetas = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=etiquetas[0])
    rec = recall_score(y_true, y_pred, pos_label=etiquetas[0])
    f1 = f1_score(y_true, y_pred, pos_label=etiquetas[0])
    print(f"[evaluar] CM:\n{cm}\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return {'cm': cm, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# --- Random Forest ---

def ejecutar_bosque_random(X_train, y_train, X_test, n_estimators=10, random_state=42):
    """
    Entrena y predice con RandomForestClassifier.

    Teoría:
      - Bagging: muestras bootstrap.
      - Selección aleatoria de atributos en cada nodo.
      - Votación mayoritaria para clasificación.
    """
    modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    print(f"[ejecutar_bosque_random] Predicciones generadas para {len(preds)} instancias.")
    return preds


def graficar_curva_precision(X_train, y_train, X_test, y_test, max_arboles=10):
    """
    Grafica precisión vs número de árboles para train y test.

    Teoría:
      - Evaluación de sobreajuste: alta precisión en train y baja en test.
      - Rendimientos decrecientes: más árboles → menor ganancia.
    """
    arboles = list(range(1, max_arboles+1))
    p_train, p_test = [], []
    etiqueta = y_train.mode()[0]
    for n in arboles:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        p_train.append(precision_score(y_train, rf.predict(X_train), pos_label=etiqueta))
        p_test.append(precision_score(y_test, rf.predict(X_test), pos_label=etiqueta))
    plt.plot(arboles, p_train, marker='o', label='Train')
    plt.plot(arboles, p_test, marker='o', label='Test')
    plt.xlabel('Número de árboles')
    plt.ylabel('Precisión')
    plt.title('Precisión vs tamaño del bosque')
    plt.legend()
    plt.show()

# --- Visualización de Árbol ID3 ---

def visualizar_arbol(arbol, nombre_archivo='arbol'):
    """
    Guarda una representación gráfica del árbol ID3 usando graphviz.

    Teoría:
      - Explicabilidad: árboles permiten interpretar decisiones.
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
if __name__ == '__main__':
    URL = 'https://drive.google.com/uc?export=download&id=1BQEFonHa5aYO4MTg1EWxGIuRSgCw6ZXb'
    COLS = [0,1,2,5,12,13]
    TARGET = 'Estado'

    # --- Pipeline completo ---
    df = cargar_datos(URL, COLS)
    df = filtrar_edad(df, 'Edad', 40, 45)
    df = preprocesar_categoricas(df, TARGET)
    train, test = dividir_entrenamiento_prueba(df)

    # ID3
    caracteristicas = [c for c in train.columns if c != TARGET]
    por_defecto = train[TARGET].mode()[0]
    arbol_id3 = construir_id3(train, TARGET, caracteristicas)
    y_pred_id3 = [predecir_id3(arbol_id3, fila, por_defecto) for _, fila in test.iterrows()]
    resultados_id3 = evaluar(test[TARGET].tolist(), y_pred_id3)
    visualizar_arbol(arbol_id3)

    # Random Forest
    print(f"\n--- Predicción Random Forest ---")
    X_tr = pd.get_dummies(train.drop(columns=[TARGET]))
    X_te = pd.get_dummies(test.drop(columns=[TARGET]))
    X_train, X_test = X_tr.align(X_te, join='left', axis=1, fill_value=0)
    y_pred_rf = ejecutar_bosque_random(X_train, train[TARGET], X_test)
    resultados_rf = evaluar(test[TARGET].tolist(), y_pred_rf)
    graficar_curva_precision(X_train, train[TARGET], X_test, test[TARGET])
