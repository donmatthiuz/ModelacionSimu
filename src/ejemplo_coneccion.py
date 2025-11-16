import numpy as np
from modelo import TracerPDE3D
from rk4 import RK4Integrator

# -------------------------------
# 1) Definir dominio espacial
# -------------------------------
nx = 50
Lx = 10.0     # metros
dx = Lx / nx
x = np.linspace(0, Lx, nx)

# campo inicial
A = np.zeros(nx)

# -------------------------------
# 2) Definir funciones de la PDE
# -------------------------------

def my_wind(x, y, z, t):
    return (1.0, 0.0, 0.0)   # viento constante de +1 m/s

def my_diff(x, y, z, t):
    return 0.0               # sin difusión

def my_source(xi, y, z, t):
    # fuente puntual en el punto x ~ 2.5 m
    return 1.0 if abs(xi - 2.5) < dx else 0.0

# Crear PDE
pde = TracerPDE3D(
    wind_field=my_wind,
    diffusivity=my_diff,
    decay_constant=0.0,
    source_function=my_source
)

# -------------------------------
# 3) Discretización espacial
# -------------------------------

def compute_F_for_all_grid(pde, A, t):
    F = np.zeros_like(A)

    # derivadas espaciales (1D)
    dA_dx = np.zeros_like(A)
    lapA = np.zeros_like(A)

    # condiciones de frontera simples (Neumann)
    dA_dx[0]  = (A[1] - A[0]) / dx
    dA_dx[-1] = (A[-1] - A[-2]) / dx

    lapA[0]   = (A[1] - 2*A[0] + A[1]) / dx**2
    lapA[-1]  = (A[-2] - 2*A[-1] + A[-2]) / dx**2

    # interior
    for i in range(1, nx-1):
        dA_dx[i] = (A[i+1] - A[i-1]) / (2*dx)
        lapA[i]  = (A[i+1] - 2*A[i] + A[i-1]) / dx**2

    # evaluar la PDE en cada punto
    for i in range(nx):
        F[i] = pde.pde_rhs(
            A[i],
            (dA_dx[i], 0, 0),
            lapA[i],
            x[i], 0, 0,
            t
        )

    return F

# -------------------------------
# 4) Integrador RK4
# -------------------------------
dt = 0.01
rk = RK4Integrator(rhs_function=lambda A,t: compute_F_for_all_grid(pde, A, t),
                   dt=dt)

# -------------------------------
# 5) Ejecutar simulación
# -------------------------------
t = 0.0
for step in range(200):
    A = rk.step(A, t)
    t += dt

print("Simulación completada.")
print("A(x) final:", A)
