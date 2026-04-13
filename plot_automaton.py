import numpy as np
from utils import randvec01
from ac import ac
import matplotlib.pyplot as plt

#
#
#
#
#
if __name__ == "__main__":
    m = 5
    n = 4
    l = 0

    for k in range(m*n*2):
        I1 = randvec01(1, 164, 0.5)[0]
        N = np.zeros((128, 164))
        a = np.zeros((2, 164))
        I = np.empty(2, dtype=object)
        
        for i in range(1, 128):
            a[0, :] = N[i-1, :]
            a[1, :] = np.roll(a[0, :], -1)
            aux1 = np.concatenate([a[:, 163:], a[:, :163]], axis=1)
            for kk in range(2):
                I[kk] = aux1[kk, :]
            ind = np.ravel_multi_index([I[0].astype(int), I[1].astype(int)], [2, 2])
            N[i, :] = k if k < 256 else 0
        
        a_mod = k % (m*n)
        if a_mod == 0:
            l = l + 1
            fig = plt.figure(num=f'AC{l}')
        
        ax = plt.subplot(m, n, a_mod + 1)
        plt.imshow(N, cmap='gray')
        ax.set_title(f'R={k}')
        ax.axis('off')

    plt.show()
