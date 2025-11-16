import numpy as np

class RK4Integrator:
    """
    Integrador de Runge-Kutta de 4to orden para sistemas:
    
        dA/dt = F(A, t)

    El integrador NO calcula F.
    El usuario debe darle una función rhs(A, t) que devuelva F(A,t)
    ---------------------------------------------------------------
    A : ndarray en 1D/2D/3D (cualquier geometría)
    rhs : callable(A, t) -> array del mismo tamaño que A
    dt : paso de tiempo
    """

    def __init__(self, rhs_function, dt):
        self.rhs = rhs_function
        self.dt = dt

    def step(self, A, t):
        """
        Realiza un paso RK4:
        
            k1 = f(A, t)
            k2 = f(A + dt/2 k1, t + dt/2)
            k3 = f(A + dt/2 k2, t + dt/2)
            k4 = f(A + dt k3,   t + dt)

            A_{n+1} = A + dt/6 (k1 + 2k2 + 2k3 + k4)
        """

        dt = self.dt
        
        k1 = self.rhs(A, t)
        k2 = self.rhs(A + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.rhs(A + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.rhs(A + dt * k3,       t + dt)

        return A + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
