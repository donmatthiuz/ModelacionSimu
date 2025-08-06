
import numpy as np

def biseccion(a, b, funcion, iter, tol):
    
    if funcion(a)*funcion(b) >= 0:
        print("Error, en el intervalo inicial debe haber un cambio de signo")
        return None, [], []
    aproximaciones = []
    yaproximaciones = []
    
    for i in range(iter+1):
        x = (a+b)/2
        aproximaciones.append(x)
        yaproximaciones.append(funcion(x))
        if funcion(x) == 0 or np.abs(a-b) < tol:
            print(f"Iteración {i+1}: raíz aproximada encontrada en x = {x}")
            return x, aproximaciones, yaproximaciones
        
        if funcion(a)*funcion(x) < 0:
            b = x
        else:
            a =x

    print("No se encontró una raíz en el número de iteraciones dadas.")
    return None, aproximaciones, yaproximaciones


def secante(a, b, funcion, iter, tol):

    aproximaciones = []
    yaproximaciones = []
    
    for i in range(iter+1):

        if funcion(a) == funcion(b):
            print("Error: división por 0")
            return None, [], []
        

        nuevo_valor = (((a*funcion(b)) - (b*funcion(a)))/(funcion(b)-funcion(a)))
        aproximaciones.append(nuevo_valor)
        yaproximaciones.append(funcion(nuevo_valor))

        if funcion(nuevo_valor) == 0 or np.abs(b-nuevo_valor) < tol:
            print(f"Iteración {i}, raíz aroximada encontrada en: {nuevo_valor}")
            return nuevo_valor, aproximaciones, yaproximaciones

        a = b
        b = nuevo_valor

    print("No se halló la raíz")
    return None, aproximaciones, yaproximaciones


def newton(a, funcion, derivada, iter, tol):
    aproximaciones = []
    yaproximaciones = []
    for i in range(iter+1):

        if derivada(a) == 0:
            print("Error: división por 0")
            return None, [], []
        
        a = a - (funcion(a)/derivada(a))
        aproximaciones.append(a)
        yaproximaciones.append(funcion(a))

        if funcion(a) == 0 or np.abs(funcion(a)-0) < tol:
            print(f"Iteración {i}, valor aproximado: {a}")
            return a, aproximaciones, yaproximaciones
        
