import numpy as np

def shannon_entropy(mat, d):
    #
    #    mat := mxn matrix
    #    d := lado del cuadrado que vamos a estudiar en cada iteración
    #
    #
    # OJO :
    #    - Hay que hacer un if para d par o impar
    #

    L, W = mat.shape
    dd = d * d

    MAT = np.zeros((L, W))
    aux = 2.0 ** np.arange(dd - 1, -1, -1)

    if d <= 0:
        raise ValueError('\n d should be a positive interger \n')

    for i in range(d - 1, L):
        for j in range(W + 1 - d):
            M = mat[(i - d + 1):(i + 1), j:(j + d)]
            M_reshaped = M.reshape(1, dd)
            MAT[i, j] = np.sum(M_reshaped * aux)

    val = np.unique(MAT)
    num_val = len(val)
    num_val_tot = np.prod(MAT.shape)
    P = np.zeros(num_val)

    for k in range(len(val)):
        P[k] = np.sum(mat == val[k]) / num_val_tot

    H = -np.sum(P * np.log2(P + 1e-10))

    return H
