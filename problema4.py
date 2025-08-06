
import numpy as np

def biseccion(a, b, funcion, iter, tol):
    
    if funcion(a)*funcion(b) >= 0:
        print("Error, en el intervalo inicial debe haber un cambio de signo")
        return None
    

    for i in range(iter):
        x = (a+b)/2

        if funcion(x) == 0 or np.abs(a-b) < tol:
            print(f"Iteración {i+1}: raíz aproximada encontrada en x = {x}")
            return x
        
        if funcion(a)*funcion(x) < 0:
            b = x
        else:
            a =x

    print("No se encontró una raíz en el número de iteraciones dadas.")
    return None