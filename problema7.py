# Problema 7
from problema4 import newton, biseccion
import numpy as np
import matplotlib.pyplot as plt

# funcion 
def f7(x):
    return x**3 - 2*x + 2

# derivada
def df7(x):
    return 3*x**2 - 2

raiz, aprox, yaprox = newton(0.0, f7, df7, iter=20, tol=1e-7)

x_vals = np.linspace(-3, 2, 1000)
y_vals = f7(x_vals)

plt.axhline(0, color='gray', linestyle='--')
plt.plot(x_vals, y_vals)
plt.title("f(x) = x^3 - 2x + 2")
plt.grid(True)
plt.show()

raiz_biseccion, _, _ = biseccion(-2.0, -1.0, f7, iter=100, tol=1e-7)

