# DEFINIR TODOS MÉTODOS

import numpy as np


def funcao(a,b,c,x):
    return a*(x**2) + b*x + c

def bisseccao(ini, fim, tolerancia, a, b, c):
    if funcao(a,b,c,ini) * funcao(a,b,c,fim) >= 0:
        raise ValueError("Não há raiz no intervalo [a, b].")
    pontos = []
    while abs(fim - ini) > tolerancia:
        meio = (ini + fim) / 2.0
        pontos.append(meio)

        if funcao(a,b,c,ini) * funcao(a,b,c,meio) < 0:
            fim = meio
        else:
            ini = meio

    return meio, pontos



