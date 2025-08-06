from problema4 import biseccion, secante, newton

def funcion(x):
    return (x**2 + 1)/(x-7)
    

def derivada(x):
    return ((x**2-(14*x) -1)/((x-7))**2)







newton(25,funcion, derivada, 100, 1e-7)