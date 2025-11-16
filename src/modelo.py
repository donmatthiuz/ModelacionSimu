import numpy as np

class TracerPDE3D:
    """
    Modelo 3-D de transporte atmosférico basado en la ecuación:
    
        ∂A/∂t = 
            - (u ∂A/∂x + v ∂A/∂y + w ∂A/∂z)       # advección
            + ∇ · (K ∇A)                          # difusión turbulenta
            - ∂(v_s A)/∂z                         # sedimentación / deposición seca
            - λ A                                 # decaimiento radiactivo
            + S(x,y,z,t)                          # fuente
    
    A = concentración o actividad (Bq/m3 o Bq/kg)
    u,v,w = campo de viento 3D
    K = coeficiente de difusión (puede ser tensorial)
    v_s = velocidad de sedimentación (si existe)
    λ = constante de decaimiento radiactivo
    S = fuente
    
    Esta clase define el sistema de EDPs pero no el método numérico.
    """

    def __init__(self, 
                 wind_field,      # función (x,y,z,t)->(u,v,w)
                 diffusivity,     # función (x,y,z,t)->K
                 decay_constant,  # λ
                 settling_velocity=0.0,  # v_s
                 source_function=None):   # S(x,y,z,t)

        self.wind_field = wind_field
        self.diffusivity = diffusivity
        self.lambda_decay = decay_constant
        self.vs = settling_velocity
        self.source = source_function if source_function else (lambda x,y,z,t: 0.0)

    def advection_term(self, A, gradA, wind):
        """
        Término advectivo: - (u Ax + v Ay + w Az)
        gradA = (dA/dx, dA/dy, dA/dz)
        """
        u, v, w = wind
        Ax, Ay, Az = gradA
        return -(u * Ax + v * Ay + w * Az)

    def diffusion_term(self, A, lapA, K):
        """
        Difusión turbulenta: ∇ · (K ∇A)  ≈ K ∇²A
        lapA = ∇²A
        """
        return K * lapA

    def settling_term(self, A, dA_dz):
        """
        Sedimentación gravitacional:
        - ∂(v_s A)/∂z = -v_s * ∂A/∂z
        """
        return -self.vs * dA_dz

    def decay_term(self, A):
        """
        Decaimiento radiactivo: -λ A
        """
        return -self.lambda_decay * A

    def source_term(self, x, y, z, t):
        """
        Fuente S(x,y,z,t)
        """
        return self.source(x, y, z, t)

    def pde_rhs(self, A, gradA, lapA, x, y, z, t):
        """
        Evalúa el lado derecho de la EDP para un punto.
        Esta función será usada por el solver numérico.

        A      = valor local
        gradA  = (dA/dx, dA/dy, dA/dz)
        lapA   = laplaciano ∇²A
        (x,y,z,t) = ubicación en el dominio
        """

        u, v, w = self.wind_field(x, y, z, t)
        K = self.diffusivity(x, y, z, t)

        dA_dt = (
            self.advection_term(A, gradA, (u, v, w)) +
            self.diffusion_term(A, lapA, K) +
            self.settling_term(A, gradA[2]) +
            self.decay_term(A) +
            self.source_term(x, y, z, t)
        )

        return dA_dt
