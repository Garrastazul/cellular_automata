import numpy as np

def ac(R, r, I1, t):
    #  ACp2 optimizado mediante vectorización de operaciones
    #
    #

    AN = len(I1)
    M = np.zeros((t, AN + 2*r))
    M[0, :] = np.concatenate([I1[(AN-r):], I1, I1[:r]])
    exp = 2.0 ** np.arange(2*r, -1, -1)
    
    # Iniciamos el algoritmo
    for i in range(1, t):
        for j in range(r, AN + r):
            calc = np.sum(M[i-1, (j-r):(j+r+1)] * exp)
            M[i, j] = R[len(R) - 1 - int(calc)]
        M[i, :r] = M[i, (AN):(AN+r)]
        M[i, (AN+r):] = M[i, r:(2*r)]
    
    N = M[:, r:(AN+r)]
    return N
