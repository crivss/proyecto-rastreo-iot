
import numpy as np


# ==========================================
# 1. METODO DE LA SECANTE
# ==========================================

def metodo_secante(f, n0, n1, tolerancia=1e-4, max_iteraciones=50):
    historial = []

    f0 = f(n0)
    f1 = f(n1)

    for k in range(1, max_iteraciones + 1):
        denominador = f1 - f0

        if abs(denominador) < 1e-12:
            raise ValueError("No se puede continuar con la secante.")

        n2 = n1 - f1 * (n1 - n0) / denominador
        f2 = f(n2)
        error = abs(n2 - n1)

        historial.append({
            "iteracion": k,
            "n": n2,
            "f_n": f2,
            "error_absoluto": error
        })

        if error < tolerancia:
            return n2, historial, True

        n0, n1 = n1, n2
        f0, f1 = f1, f2

    return n1, historial, False


# ==========================================
# 2. METODO DE GAUSS-SEIDEL
# ==========================================

def gauss_seidel(matriz, vector, semilla=(5.0, 5.0),
                 tolerancia=1e-5, max_iteraciones=1000):

    matriz = np.asarray(matriz, dtype=float)
    vector = np.asarray(vector, dtype=float)
    solucion = np.asarray(semilla, dtype=float).copy()

    n = len(vector)

    if matriz.shape != (n, n):
        raise ValueError("La matriz debe ser cuadrada.")

    if solucion.shape != (n,):
        raise ValueError("La semilla no tiene la dimensión correcta.")

    if not np.all(np.isfinite(matriz)) or not np.all(np.isfinite(vector)):
        raise ValueError("La matriz y el vector deben contener valores finitos.")

    if np.any(np.abs(np.diag(matriz)) < 1e-12):
        raise ValueError("La diagonal contiene un valor nulo.")

    historial = []
    convergio = False

    for k in range(1, max_iteraciones + 1):
        anterior = solucion.copy()

        for i in range(n):
            suma_izquierda = np.dot(
                matriz[i, :i], solucion[:i]
            )

            suma_derecha = np.dot(
                matriz[i, i + 1:], anterior[i + 1:]
            )

            solucion[i] = (
                vector[i] - suma_izquierda - suma_derecha
            ) / matriz[i, i]

        diferencia = np.linalg.norm(solucion - anterior)
        magnitud = np.linalg.norm(solucion)

        if magnitud > 0:
            error_relativo = diferencia / magnitud
        else:
            error_relativo = diferencia

        historial.append({
            "iteracion": k,
            "x": solucion[0],
            "y": solucion[1],
            "error_relativo": error_relativo
        })

        if error_relativo < tolerancia:
            convergio = True
            break

    return solucion, historial, convergio


# ==========================================
# 3. TRILATERACION Y ANALISIS DE CONDICION
# ==========================================

def trilateracion_gauss_seidel(
        anclas, distancias, semilla=(5.0, 5.0),
        tolerancia=1e-5, max_iteraciones=1000):

    anclas = np.asarray(anclas, dtype=float)
    distancias = np.asarray(distancias, dtype=float)

    if anclas.ndim != 2 or anclas.shape[1] != 2:
        raise ValueError("Las anclas deben tener coordenadas X e Y.")

    if len(anclas) < 3 or len(distancias) != len(anclas):
        raise ValueError("Se necesitan al menos tres anclas y sus distancias.")

    if not np.all(np.isfinite(anclas)) or not np.all(np.isfinite(distancias)):
        raise ValueError("Las coordenadas y distancias deben ser finitas.")

    if np.any(distancias < 0):
        raise ValueError("Las distancias no pueden ser negativas.")

    # Primera ancla como referencia
    x1, y1 = anclas[0]
    d1 = distancias[0]

    A = []
    b = []

    for i in range(1, len(anclas)):
        xi, yi = anclas[i]
        di = distancias[i]

        A.append([
            2 * (xi - x1),
            2 * (yi - y1)
        ])

        b.append(
            d1**2 - di**2
            + xi**2 - x1**2
            + yi**2 - y1**2
        )

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("La distribución de anclas no permite localizar en 2D.")

    # Sistema de ecuaciones normales: M X = C
    M = A.T @ A
    C = A.T @ b

    # Numero de condicion de la matriz
    condicion = np.linalg.cond(M)

    # Resolver el sistema mediante Gauss-Seidel
    posicion, historial, convergio = gauss_seidel(
        M,
        C,
        semilla=semilla,
        tolerancia=tolerancia,
        max_iteraciones=max_iteraciones
    )

    return posicion, historial, convergio, condicion
