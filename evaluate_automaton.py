import numpy as np
from utils import randvec01

ancho = 301
t = 2*ancho
r = 3
d_cambio = 2*r + 1
numCI = int(1e4)
mat_evaluacion = np.zeros((numCI, ancho))

def CondicionesFinales(num_CI, ancho):
    porcentajes_iniciales = np.zeros(num_CI)
    A = np.zeros((num_CI, ancho))
    
    for a in range(num_CI):
        A[a, :] = randvec01(1, ancho, a/(num_CI+1))[0]
        porcentajes_iniciales[a] = np.mean(A[a, :])
    
    Cf = np.round(porcentajes_iniciales, 0)
    return A, Cf

if __name__ == "__main__":
    A, Cf = CondicionesFinales(numCI, ancho)

    for mat in range(numCI):
        print(f'iter: {mat:f}')
