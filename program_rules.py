import numpy as np

#
#   Script para programar reglas más facilmente
#
#   R = [a_{N}, a_{N-1}, ..., a_{1}, a_{0}], donde a_{0} es el cambio para
#       el 0 en binario
#

if __name__ == "__main__":
    r = 1
    d_cambio = 1 + 2*r
    R = np.zeros(2**d_cambio)

    print('\n' + '='*66)
    print('\n    Script para facilitar la programación de reglas para AC     \n')
    print('='*66 + '\n')

    for i in range(2**d_cambio):
        a = bin(2**d_cambio - 1 - i)[2:].zfill(d_cambio)
        while True:
            try:
                value = int(input(f'\n {a} --valor-->'))
                if value not in [0, 1]:
                    print('El valor introducido no es 0 o 1 :(')
                else:
                    R[i] = value
                    break
            except ValueError:
                print('El valor introducido no es 0 o 1 :(')
