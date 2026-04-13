"""
Funciones Auxiliares para Autómatas Celulares y Algoritmo Genético
"""
import numpy as np


def randvec01(filas, columnas, P0):
    """
    Genera matriz binaria aleatoria
    
    Parameters:
        filas, columnas: dimensiones
        P0: probabilidad de 0 (1-P0 es la probabilidad de 1)
        
    Returns:
        matriz binaria (filas × columnas)
    """
    A = np.zeros((filas, columnas))
    for i in range(filas):
        for j in range(columnas):
            A[i, j] = 1 if np.random.rand() > P0 else 0
    
    return A


def CondicionesFinales(num_CI, ancho):
    """
    Genera condiciones iniciales y objetivos finales
    
    Objetivo: que el autómata converja a todo 0s o todo 1s
    según la densidad inicial.
    
    Parameters:
        num_CI: número de condiciones iniciales
        ancho: ancho del autómata
    
    Returns:
        A: matriz de condiciones iniciales (num_CI × ancho)
        Cf: matriz de estados objetivo (num_CI × ancho)
    """
    A = np.zeros((num_CI, ancho))
    Cf = np.zeros((num_CI, ancho))
    
    for a in range(num_CI):
        # Condición inicial con densidad uniforme
        A[a, :] = randvec01(1, ancho, a / (num_CI + 1))[0]
        densidad = np.mean(A[a, :])
        
        # Objetivo: todo 0s si densidad <= 0.5, todo 1s si > 0.5
        Cf[a, :] = np.ones(ancho) if densidad > 0.5 else np.zeros(ancho)
    
    return A, Cf


def callR(nom, r):
    """
    Convierte número decimal a representación binaria de regla
    
    Parameters:
        nom: número en decimal
        r: radio de vecindad
        
    Returns:
        array binario con bit_length = 2^(2*r+1) elementos
    """
    bit_length = 2 ** (2 * r + 1)
    R = np.array([(int(nom) >> (bit_length - 1 - i)) & 1 for i in range(bit_length)])
    return R


def calpercent(A):
    """
    Calcula densidad (porcentaje de 1s) por fila
    
    Parameters:
        A: matriz binaria
        
    Returns:
        array con densidad por cada fila
    """
    A = np.array(A)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    filas, columnas = A.shape
    porcentajes = np.zeros(filas)
    
    for i in range(filas):
        porcentajes[i] = np.sum(A[i, :]) / columnas
    
    return porcentajes


def numcoinc(A, B):
    """
    Cuenta coincidencias elemento a elemento entre dos arrays
    
    Parameters:
        A, B: arrays (serán aplanados)
        
    Returns:
        número de coincidencias
    """
    A = np.array(A).flatten()
    B = np.array(B).flatten()
    return np.sum(A == B)


def matcolon(A, B):
    """
    Genera listas de rango para cada par (A[k], B[k])
    
    Parameters:
        A, B: arrays de inicio y fin
        
    Returns:
        lista de listas con rangos
    """
    lA = len(A)
    a = [None] * lA
    
    for k in range(lA):
        a[k] = list(range(int(A[k]), int(B[k]) + 1))
    
    return a


def shannon_entropy(mat, d=2):
    """
    Calcula entropía de Shannon en ventanas locales d×d
    
    Parameters:
        mat: matriz binaria
        d: tamaño de ventana (default: 2)
        
    Returns:
        valor de entropía en bits
    """
    L, W = mat.shape
    dd = d * d
    aux = 2.0 ** np.arange(dd - 1, -1, -1)
    
    MAT = np.zeros((L, W))
    for i in range(d - 1, L):
        for j in range(W + 1 - d):
            M = mat[(i - d + 1):(i + 1), j:(j + d)]
            MAT[i, j] = np.sum(M.reshape(1, dd) * aux)
    
    val = np.unique(MAT)
    P = np.array([np.sum(MAT == v) / np.prod(MAT.shape) for v in val])
    H = -np.sum(P * np.log2(P + 1e-10))
    
    return H


if __name__ == "__main__":
    print("Funciones auxiliares cargadas correctamente")
    print("\nFunciones disponibles:")
    print("  * randvec01() - Matriz binaria aleatoria")
    print("  * CondicionesFinales() - Condiciones iniciales y objetivos")
    print("  * callR() - Decimal a binario de regla")
    print("  * calpercent() - Densidad por fila")
    print("  * numcoinc() - Coincidencias entre arrays")
    print("  * matcolon() - Generador de rangos")
    print("  * shannon_entropy() - Entropia de Shannon")
