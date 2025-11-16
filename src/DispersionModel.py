import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class ChernobylDispersionModel:
    def __init__(self, nx=100, ny=100, nz=20, dx=50000, dy=50000, dz=500, dt=3600):
        """
        Inicialización del modelo
        
        Parámetros:
        nx, ny, nz: número de puntos en x, y, z
        dx, dy, dz: espaciamiento de malla (metros)
        dt: paso temporal (segundos)
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        
        # Campo de concentración (x, y, z)
        self.C = np.zeros((nx, ny, nz))
        
        # Parámetros físicos
        self.Kh = 1000.0  # Difusividad horizontal (m²/s)
        self.Kv = 10.0    # Difusividad vertical (m²/s)
        self.vd = 0.001   # Velocidad de deposición (m/s)
        self.lambda_decay = np.log(2) / (30.17 * 365 * 24 * 3600)  # Cs-137
        
        # Posición de la fuente (Chernobyl - centro del dominio)
        self.source_x = nx // 2
        self.source_y = ny // 2
        self.source_z_min = 4  # ~200m altura
        self.source_z_max = 30 # ~1500m altura
        
        # Campo de viento (m/s)
        self.u = np.zeros((nx, ny, nz))  # componente x
        self.v = np.zeros((nx, ny, nz))  # componente y
        self.w = np.zeros((nx, ny, nz))  # componente z
        
        # Tasa de emisión inicial (Bq/s)
        self.Q0 = 1e15
        
        # Tiempo de simulación
        self.time = 0
        self.time_hours = 0
        
        # Deposición acumulada
        self.deposition = np.zeros((nx, ny))
        
    def set_wind_field(self, u_mean=5.0, v_mean=3.0, variability=2.0):
        # Viento zonal (oeste-este)
        self.u = u_mean + variability * np.random.randn(self.nx, self.ny, self.nz) * 0.3
        
        # Viento meridional (sur-norte)
        self.v = v_mean + variability * np.random.randn(self.nx, self.ny, self.nz) * 0.3
        
        # Viento vertical (pequeño)
        self.w = 0.01 * np.random.randn(self.nx, self.ny, self.nz)
        
        # Suavizar campos de viento
        for k in range(self.nz):
            self.u[:,:,k] = gaussian_filter(self.u[:,:,k], sigma=2)
            self.v[:,:,k] = gaussian_filter(self.v[:,:,k], sigma=2)
            self.w[:,:,k] = gaussian_filter(self.w[:,:,k], sigma=2)
    
    def emission_rate(self, t):
        # Emisión activa durante ~10 días
        t_days = t / (24 * 3600)
        if t_days < 10:
            return self.Q0 * np.exp(-t_days / 5.0)
        else:
            return 0.0
    
    def add_source(self):
        Q = self.emission_rate(self.time)
        
        # Distribuir emisión verticalmente (normalizando pesos)
        k_start = int(self.source_z_min)
        k_end = int(min(self.source_z_max, self.nz))
        ks = list(range(k_start, k_end))
        if len(ks) == 0 or Q == 0.0:
            return
        weights = np.array([np.exp(-(k - self.source_z_min)**2 / 50.0) for k in ks], dtype=float)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights_sum
        source_vol = self.dx * self.dy * self.dz
        for idx, k in enumerate(ks):
            source_term = Q * weights[idx] / source_vol
            self.C[self.source_x, self.source_y, k] += source_term * self.dt
    
    def advection_step(self):
        C_new = self.C.copy()
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    # Advección en x
                    if self.u[i,j,k] > 0:
                        flux_x = self.u[i,j,k] * (self.C[i,j,k] - self.C[i-1,j,k]) / self.dx
                    else:
                        flux_x = self.u[i,j,k] * (self.C[i+1,j,k] - self.C[i,j,k]) / self.dx
                    
                    # Advección en y
                    if self.v[i,j,k] > 0:
                        flux_y = self.v[i,j,k] * (self.C[i,j,k] - self.C[i,j-1,k]) / self.dy
                    else:
                        flux_y = self.v[i,j,k] * (self.C[i,j+1,k] - self.C[i,j,k]) / self.dy
                    
                    # Advección en z
                    if self.w[i,j,k] > 0:
                        flux_z = self.w[i,j,k] * (self.C[i,j,k] - self.C[i,j,k-1]) / self.dz
                    else:
                        if k < self.nz-1:
                            flux_z = self.w[i,j,k] * (self.C[i,j,k+1] - self.C[i,j,k]) / self.dz
                        else:
                            flux_z = 0
                    
                    C_new[i,j,k] -= self.dt * (flux_x + flux_y + flux_z)
        
        self.C = np.maximum(C_new, 0)  # Concentraciones no negativas
    
    def diffusion_step(self):
        C_new = self.C.copy()
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    # Difusión horizontal
                    laplacian_h = (self.C[i+1,j,k] - 2*self.C[i,j,k] + self.C[i-1,j,k]) / self.dx**2 + \
                                  (self.C[i,j+1,k] - 2*self.C[i,j,k] + self.C[i,j-1,k]) / self.dy**2
                    
                    # Difusión vertical
                    laplacian_v = (self.C[i,j,k+1] - 2*self.C[i,j,k] + self.C[i,j,k-1]) / self.dz**2
                    
                    C_new[i,j,k] += self.dt * (self.Kh * laplacian_h + self.Kv * laplacian_v)
        
        self.C = np.maximum(C_new, 0)
    
    def deposition_step(self):
        # Deposición seca - una capa 
        deposition_flux = self.vd * self.C[:, :, 0]
        self.deposition += deposition_flux * self.dt
        self.C[:, :, 0] -= (self.vd / self.dz) * self.C[:, :, 0] * self.dt
        
        # Decaimiento radiactivo
        self.C *= np.exp(-self.lambda_decay * self.dt)
    
    def apply_boundary_conditions(self):
        # Fronteras laterales: concentración cero
        self.C[0, :, :] = 0
        self.C[-1, :, :] = 0
        self.C[:, 0, :] = 0
        self.C[:, -1, :] = 0
        
        # Frontera superior: gradiente cero
        self.C[:, :, -1] = self.C[:, :, -2]
        
        # Frontera inferior: deposición
        # (ya manejado en deposition_step)
    
    def step(self):
        # Añadir emisión
        self.add_source()
        
        # Advección
        self.advection_step()
        
        # Difusión
        self.diffusion_step()
        
        # Deposición y decaimiento
        self.deposition_step()
        
        # Condiciones de frontera
        self.apply_boundary_conditions()
        
        # Actualizar tiempo
        self.time += self.dt
        self.time_hours = self.time / 3600
    
    def get_column_integrated(self):
        return np.sum(self.C, axis=2) * self.dz
    
    def run_simulation(self, num_steps=240):
        print(f"Iniciando simulación de dispersión de Chernobyl...")
        print(f"Pasos de tiempo: {num_steps}")
        print(f"Tiempo total: {num_steps * self.dt / 3600:.1f} horas")
        
        concentrations = []
        depositions = []
        
        for step in range(num_steps):
            self.step()
            
            if step % 10 == 0:
                concentrations.append(self.get_column_integrated().copy())
                depositions.append(self.deposition.copy())
                print(f"Paso {step}/{num_steps} - Tiempo: {self.time_hours:.1f} h")
        
        return concentrations, depositions


# visualizar resultados
def visualize_results(model, concentrations, depositions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Concentración integrada en columna 
    ax1 = axes[0, 0]
    c1 = concentrations[-1]
    im1 = ax1.imshow(c1.T, origin='lower', cmap='hot', 
                     norm=plt.Normalize(vmin=0, vmax=np.percentile(c1, 99)))
    ax1.set_title('Concentración Integrada en Columna (final)')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    plt.colorbar(im1, ax=ax1, label='Bq·m/m²')
    
    # Deposición acumulada
    ax2 = axes[0, 1]
    dep = depositions[-1]
    im2 = ax2.imshow(dep.T, origin='lower', cmap='YlOrRd',
                     norm=plt.Normalize(vmin=0, vmax=np.percentile(dep, 99)))
    ax2.set_title('Deposición Acumulada Total')
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    plt.colorbar(im2, ax=ax2, label='Bq/m²')
    
    # Perfil vertical en la fuente
    ax3 = axes[1, 0]
    vertical_profile = model.C[model.source_x, model.source_y, :]
    heights = np.arange(model.nz) * model.dz / 1000  # km
    ax3.plot(vertical_profile, heights, 'b-', linewidth=2)
    ax3.set_xlabel('Concentración (Bq/m³)')
    ax3.set_ylabel('Altura (km)')
    ax3.set_title('Perfil Vertical en Chernobyl')
    ax3.grid(True, alpha=0.3)
    
    # Evolución temporal de masa total
    ax4 = axes[1, 1]
    total_mass = [np.sum(c) * model.dx * model.dy for c in concentrations]
    times = np.arange(len(total_mass)) * 10 * model.dt / 3600  # horas
    ax4.plot(times, np.array(total_mass) / 1e15, 'g-', linewidth=2)
    ax4.set_xlabel('Tiempo (horas)')
    ax4.set_ylabel('Masa Total en Atmósfera (PBq)')
    ax4.set_title('Evolución de Masa Total Atmosférica')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chernobyl_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nRESULTADOS FINALES") 
    print(f"Tiempo simulado: {model.time_hours:.1f} horas")
    print(f"Concentración máxima: {np.max(model.C):.2e} Bq/m³")
    print(f"Deposición máxima: {np.max(model.deposition):.2e} Bq/m²")
    print(f"Deposición total: {np.sum(model.deposition) * model.dx * model.dy / 1e15:.2f} PBq")


print("SIMULACIÓN DE DISPERSIÓN ATMOSFÉRICA - CHERNOBYL 1986")
    
# modelo (ahorita tiene parametros para que funcione y se pruebe rapido) 
model = ChernobylDispersionModel(
    nx=80,      # puntos en x
    ny=80,      # puntos en y
    nz=15,      # niveles verticales
    dx=60000,   # 60 km
    dy=60000,   # 60 km
    dz=300,     # 300 m
    dt=1800     # 30 minutos
)
 


print("\nConfigurando campo de viento...")
model.set_wind_field(u_mean=6.0, v_mean=4.0, variability=2.5)
concentrations, depositions = model.run_simulation(num_steps=200)
visualize_results(model, concentrations, depositions)    

