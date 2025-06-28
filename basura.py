# def construir_id3(df, target, atributos, profundidad_max=None, profundidad_actual=0):
#     """
#     Construye recursivamente un árbol de decisión usando ID3 con límite de profundidad.
#     """
#     # Caso base: nodo puro
#     if len(df[target].unique()) == 1:
#         clase = df[target].iloc[0]
#         return clase

#     # Caso base: sin atributos o se alcanzó la profundidad máxima
#     if not atributos or (profundidad_max is not None and profundidad_actual >= profundidad_max):
#         moda = df[target].mode()[0]
#         return moda

#     # Elegir mejor atributo según ganancia de información
#     ganancias = {atributo: ganancia_informacion(df, atributo, target) for atributo in atributos}
#     atributo_mejor_ganancia = max(ganancias, key=ganancias.get)
#     arbol = {atributo_mejor_ganancia: {}}

#     for valor, sub in df.groupby(atributo_mejor_ganancia):
#         arbol[atributo_mejor_ganancia][valor] = construir_id3(
#             sub,
#             target,
#             [a for a in atributos if a != atributo_mejor_ganancia],
#             profundidad_max,
#             profundidad_actual + 1
#         )
#     return arbol