# Problema 6, aprovechando las funciones del problema 4
import matplotlib.pyplot as plt
import numpy as np
from problema4 import biseccion

# funcion dada
def f(x):
    return 2*x**5 + 3*x**4 - 3*x**3 - 10*x**2 - 4*x + 4

x_vals = np.linspace(-5, 5, 1000)
y_vals = f(x_vals)

plt.axhline(0, color='gray', linestyle='--')
plt.plot(x_vals, y_vals)
plt.title("f(x) = 2x^5 + 3x^4 - 3x^3 - 10x^2 - 4x + 4")
plt.grid(True)
plt.show()

intervalos = [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2)]

for a, b in intervalos:
    print(f"\nBuscando raíz entre {a} y {b}")
    raiz, aprox, yaprox = biseccion(a, b, f, iter=100, tol=1e-7)
    if raiz is not None:
        print(f"Raíz aproximada: {raiz}")

