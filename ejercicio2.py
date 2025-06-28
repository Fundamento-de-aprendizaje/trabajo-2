
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt

# ---------------------------------------
# FUNCIONES AUXILIARES PARA EL EJERCICIO
# ---------------------------------------

def cargar_datos(url, columnas, codificacion='latin1'):
    df = pd.read_csv(url, usecols=columnas, encoding=codificacion)
    print(f"[cargar_datos] Cargadas {len(df)} filas.")
    return df

def filtrar_edad(df, columna_edad, edad_min, edad_max):
    df_filtrado = df[(df[columna_edad] >= edad_min) & (df[columna_edad] <= edad_max)].copy()
    print(f"[filtrar_edad] Filtrado: {len(df_filtrado)} filas.")
    return df_filtrado

def preprocesar_categoricas(df, columna_target):
    df = df.select_dtypes(include='object').dropna(subset=[columna_target])
    print(f"[preprocesar_categoricas] Quedan {len(df)} filas con columnas categóricas.")
    return df

def dividir_entrenamiento_prueba(df, prueba_size=0.2, random_state=42):
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    idx = int(len(df_shuffled)*(1 - prueba_size))
    return df_shuffled.iloc[:idx], df_shuffled.iloc[idx:]

# ---------------------------------------
# FUNCIONES PARA RANDOM FOREST
# ---------------------------------------

def aplicar_random_forest(X_train, y_train, X_test, n_arboles=10):
    modelo = RandomForestClassifier(n_estimators=n_arboles, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print("[aplicar_random_forest] Predicción completada.")
    return y_pred, modelo

def mostrar_matriz_confusion(y_true, y_pred):
    print("[mostrar_matriz_confusion] Matriz de Confusión:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

def mostrar_accuracy_f1(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=y_true.mode()[0])
    print(f"[mostrar_accuracy_f1] Accuracy: {acc:.4f}")
    print(f"[mostrar_accuracy_f1] F1-Score: {f1:.4f}")

def graficar_precision_vs_tamano(X_train, y_train, X_test, y_test, max_arboles=10):
    etiquetas = sorted(list(set(y_train)))
    pos_label = y_train.mode()[0]
    train_scores = []
    test_scores = []

    for n in range(1, max_arboles + 1):
        modelo = RandomForestClassifier(n_estimators=n, random_state=42)
        modelo.fit(X_train, y_train)
        y_train_pred = modelo.predict(X_train)
        y_test_pred = modelo.predict(X_test)

        p_train = precision_score(y_train, y_train_pred, pos_label=pos_label)
        p_test = precision_score(y_test, y_test_pred, pos_label=pos_label)

        train_scores.append(p_train)
        test_scores.append(p_test)

    plt.plot(range(1, max_arboles + 1), train_scores, label='Train', marker='o')
    plt.plot(range(1, max_arboles + 1), test_scores, label='Test', marker='o')
    plt.title('Precisión vs Tamaño del Bosque')
    plt.xlabel('Número de Árboles')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------
# MAIN
# ---------------------------------------

if __name__ == "__main__":
    URL = 'https://drive.google.com/uc?export=download&id=1BQEFonHa5aYO4MTg1EWxGIuRSgCw6ZXb'
    COLUMNAS = [0,1,2,5,12,13]
    TARGET = 'Estado'

    df = cargar_datos(URL, COLUMNAS)
    df = filtrar_edad(df, 'Edad', 40, 45)
    df = preprocesar_categoricas(df, TARGET)
    df_train, df_test = dividir_entrenamiento_prueba(df)

    X_train = pd.get_dummies(df_train.drop(columns=[TARGET]))
    X_test = pd.get_dummies(df_test.drop(columns=[TARGET]))
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    y_train = df_train[TARGET]
    y_test = df_test[TARGET]

    y_pred, modelo = aplicar_random_forest(X_train, y_train, X_test)
    mostrar_matriz_confusion(y_test, y_pred)
    mostrar_accuracy_f1(y_test, y_pred)
    graficar_precision_vs_tamano(X_train, y_train, X_test, y_test)
