import numpy as np

def callR(nom, r):
    #
    #
    #  callR(nom,r) convierte el número nom (en decimal) en una matriz cuyos
    #  coeficientes son los correspondientes dígitos en binario.
    #
    #   Ej: callR(30,1)
    #

    bit_length = 2**(2*r+1)
    R = np.array([(int(nom) >> (bit_length - 1 - i)) & 1 for i in range(bit_length)])
    return R

def calpercent(A):
    # porcentaje de unos en cada columna de A

    A = np.array(A)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    filas, columnas = A.shape
    porcentajes_iniciales = np.zeros(filas)

    for i in range(filas):
        n = 0
        for j in range(columnas):
            if A[i, j] == 1:
                n = n + 1
        porcentajes_iniciales[i] = n / columnas

    return porcentajes_iniciales

def numcoinc(A, B):
    A = np.array(A).flatten()
    B = np.array(B).flatten()
    n_coincidencias = 0
    for i in range(len(A)):
        if A[i] == B[i]:
            n_coincidencias = n_coincidencias + 1

    return n_coincidencias

def randvec01(filas, columnas, P0):
    A = np.zeros((filas, columnas))
    for i in range(filas):
        for j in range(columnas):
            P = np.random.rand()
            if P < P0:
                A[i, j] = 0
            else:
                A[i, j] = 1
    
    return A

def matcolon(A, B):
    lA = len(A)
    a = [None] * lA

    for k in range(lA):
        a[k] = list(range(int(A[k]), int(B[k]) + 1))

    return a
