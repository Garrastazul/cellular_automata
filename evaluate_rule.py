import numpy as np
from utils import randvec01

ancho = 101

def CondicionesFinales(num_CI, ancho):
    porcentajes_iniciales = np.zeros(num_CI)
    A = np.zeros((num_CI, ancho))
    Cf = np.zeros((num_CI, ancho))
    
    for a in range(num_CI):
        A[a, :] = randvec01(1, ancho, 0.5)[0]
        porcentajes_iniciales[a] = np.mean(A[a, :])
        if porcentajes_iniciales[a] <= 0.5:
            Cf[a, :] = np.zeros(ancho)
        else:
            Cf[a, :] = np.ones(ancho)
    
    return A, Cf

if __name__ == "__main__":
    A, Cf = CondicionesFinales(10000, ancho)

    # La nota final de la mejor regla es 0.8020.
