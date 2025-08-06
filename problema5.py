from problema4 import biseccion

def funcion(x):
    return (x**2 + 1)/(x-7)
    
def funcion2(x):
    return (2*(x**3))+ (3*(x**2)) + x -26

print(biseccion(-1, 3, funcion2, 100, 1e-7))

print(funcion2(2))