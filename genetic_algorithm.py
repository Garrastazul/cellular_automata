import numpy as np
from utils import randvec01, numcoinc
from ac import ac
import matplotlib.pyplot as plt

#
#   ALGORITMO GENÉTICO CON AC 1D
#
#   versión final optimizada
#

def CondicionesFinales(num_CI, ancho):
    porcentajes_iniciales = np.zeros(num_CI)
    A = np.zeros((num_CI, ancho))
    Cf = np.zeros((num_CI, ancho))
    
    for a in range(num_CI):
        A[a, :] = randvec01(1, ancho, a/(num_CI+1))[0]
        porcentajes_iniciales[a] = np.mean(A[a, :])
        if porcentajes_iniciales[a] <= 0.5:
            Cf[a, :] = np.zeros(ancho)
        else:
            Cf[a, :] = np.ones(ancho)
    
    return A, Cf

r = 3
d_cambio = 2*r + 1
lR = 2**d_cambio

poblacion = 100
num_parents = 10
num_CI = 100
gen_max = 1000
ancho = 101
t = 2*ancho

np.random.seed()

mat_evaluacion = np.zeros(num_CI)
nota_reglas = np.zeros(poblacion)
mejor_nota = np.zeros(gen_max)
p_mutacion = 0.05

R_next = np.zeros((poblacion, lR))
R_mutada = np.zeros((poblacion, lR))
R_mejor = np.zeros((gen_max, lR))
Save_R = np.zeros((poblacion, lR, gen_max))

j_idx, i_idx = np.where(~np.eye(num_parents, dtype=bool))
on = np.ones(poblacion - num_parents)
lRon = lR * on

N = np.ones((t, ancho))
a = np.zeros((d_cambio, ancho))
I = np.empty(d_cambio, dtype=object)

def ac_main(R, r, I1, t, d_cambio, N, a, I):
    R = R + 1
    R = np.flipud(R)

    N[0, :] = I1 + 1
    
    for i in range(1, t):
        a[0, :] = N[i-1, :]
        for k in range(1, d_cambio):
            a[k, :] = np.roll(a[k-1, :], -1)

        aux1 = np.concatenate([a[:, (len(I1)-r):], a[:, :(len(I1)-r)]], axis=1)

        for k in range(d_cambio):
            I[k] = aux1[k, :]

        ind = np.ravel_multi_index([I[k].astype(int) for k in range(d_cambio)], [2]*d_cambio)
        N[i, :] = R[ind]

    N = N - 1
    return N

def isallequal(A, B, ancho):
    val = (np.count_nonzero(A == B) == ancho)
    return val

for g in range(gen_max):
    Save_R[:, :, g] = R
    
    A, Cf = CondicionesFinales(num_CI, ancho)

    for rule in range(poblacion):
        for mat in range(num_CI):
            N_eval = ac_main(R[rule, :].copy(), r, A[mat, :].copy(), t, d_cambio, N.copy(), a.copy(), I)
            mat_evaluacion[mat] = isallequal(N_eval[t-1, :], Cf[mat], ancho)
        
        nota_reglas[rule] = np.mean(mat_evaluacion)

    O = np.sort(nota_reglas)[::-1]
    index = np.argsort(nota_reglas)[::-1]
    mejor_nota[g] = O[0]
    R_next[0:num_parents, :] = R[index[0:num_parents], :]

    print(f'Generación: {g:f}     Nota: {O[0]:4f}')

    p_particion = np.sort(np.round((lR - 1) * np.random.rand(poblacion - num_parents, 2) + 1, 0).astype(int), axis=1)
    R_next[num_parents:, :] = R[i_idx, :]
    
    for k in range(poblacion - num_parents):
        R_next[k + num_parents, p_particion[k, 0]:p_particion[k, 1]] = R_next[j_idx[k % len(j_idx)], p_particion[k, 0]:p_particion[k, 1]]

    Mutaciones = np.random.rand(poblacion, lR) < p_mutacion
    R_mutada = np.mod(R_next + Mutaciones, 2)
    p = 1 + (poblacion - 1) * np.random.rand(poblacion - num_parents)
    R_next[num_parents:poblacion, :] = R_mutada[np.round(p, 0).astype(int), :]

    R = R_next.copy()

if __name__ == "__main__":
    plt.figure(figsize=(10, 6), num='Evolución de la población')
    plt.plot(range(1, gen_max + 1), mejor_nota, '-b.', markersize=10)
    plt.xlabel('Generaciones')
    plt.ylabel('Nota del mejor individuo')
    plt.show()
