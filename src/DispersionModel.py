import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class ChernobylDispersionModel:
    def __init__(self, nx=100, ny=100, nz=20, dx=50000, dy=50000, dz=500, dt=3600):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        
        self.C = np.zeros((nx, ny, nz))
        
        # 🔧 PARÁMETROS FÍSICOS MEJORADOS
        self.Kh = 260000.0    
        self.Kv = 5.0        
        self.vd = 0.001      
        self.lambda_decay = np.log(2) / (30.17 * 365 * 24 * 3600)
        
        self.source_x = nx // 2
        self.source_y = ny // 2
        self.source_z_min = 2
        self.source_z_max = 8
        
        self.u = np.zeros((nx, ny, nz))
        self.v = np.zeros((nx, ny, nz))
        self.w = np.zeros((nx, ny, nz))
        
        self.Q0 = 1e15
        
        self.time = 0
        self.time_hours = 0
        
        self.deposition = np.zeros((nx, ny))
        
        # 🆕 PARÁMETROS DE VIENTO VARIABLE
        self.wind_update_interval = 6 * 3600  # Cambiar viento cada 6 horas
        self.last_wind_update = 0
        
    def set_wind_field_time_varying(self, t):
        """
        🆕 Viento que rota con el tiempo para simular cambios meteorológicos
        """
        # Período de rotación del viento: ~3 días
        rotation_period = 3 * 24 * 3600  # segundos
        angle = 2 * np.pi * (t / rotation_period)
        
        # Velocidad base que varía lentamente
        base_speed = 7.0 + 3.0 * np.sin(2 * np.pi * t / (2 * 24 * 3600))
        
        u_base = base_speed * np.cos(angle)
        v_base = base_speed * np.sin(angle)
        
        # Variabilidad espacial más fuerte
        for k in range(self.nz):
            # 🔧 Perturbaciones más grandes (no *0.3)
            noise_u = 4.0 * np.random.randn(self.nx, self.ny)
            noise_v = 4.0 * np.random.randn(self.nx, self.ny)
            
            # Suavizar las perturbaciones
            smooth_u = gaussian_filter(noise_u, sigma=3)
            smooth_v = gaussian_filter(noise_v, sigma=3)
            
            # Viento varía con la altura
            height_factor = 1.0 + 0.3 * (k / self.nz)
            
            self.u[:,:,k] = (u_base + smooth_u) * height_factor
            self.v[:,:,k] = (v_base + smooth_v) * height_factor
            self.w[:,:,k] = 0.02 * np.random.randn(self.nx, self.ny)
    
    def set_wind_field(self, u_mean=5.0, v_mean=3.0, variability=2.0):
        """Método original (para compatibilidad)"""
        self.set_wind_field_time_varying(0)
    
    def emission_rate(self, t):
        t_days = t / (24 * 3600)
        if t_days < 10:
            return self.Q0 * np.exp(-t_days / 5.0)
        else:
            return 0.0
    
    def add_source(self):
        Q = self.emission_rate(self.time)
        
        k_start = max(0, int(self.source_z_min))
        k_end = min(int(self.source_z_max), self.nz)
        ks = list(range(k_start, k_end))
        if len(ks) == 0 or Q == 0.0:
            return
        
        weights = np.array([np.exp(-(k - self.source_z_min)**2 / 8.0) for k in ks], dtype=float)
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
        """Esquema upwind de primer orden"""
        C_new = self.C.copy()
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    if self.u[i,j,k] > 0:
                        dC_dx = (self.C[i,j,k] - self.C[i-1,j,k]) / self.dx
                    else:
                        dC_dx = (self.C[i+1,j,k] - self.C[i,j,k]) / self.dx
                    
                    if self.v[i,j,k] > 0:
                        dC_dy = (self.C[i,j,k] - self.C[i,j-1,k]) / self.dy
                    else:
                        dC_dy = (self.C[i,j+1,k] - self.C[i,j,k]) / self.dy
                    
                    if self.w[i,j,k] > 0:
                        dC_dz = (self.C[i,j,k] - self.C[i,j,k-1]) / self.dz
                    else:
                        if k < self.nz-1:
                            dC_dz = (self.C[i,j,k+1] - self.C[i,j,k]) / self.dz
                        else:
                            dC_dz = 0
                    
                    C_new[i,j,k] = self.C[i,j,k] - self.dt * (
                        self.u[i,j,k] * dC_dx + 
                        self.v[i,j,k] * dC_dy + 
                        self.w[i,j,k] * dC_dz
                    )
        
        self.C = np.maximum(C_new, 0)
    
    def diffusion_step(self):
        """Difusión explícita"""
        C_new = self.C.copy()
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    laplacian_h = (
                        (self.C[i+1,j,k] - 2*self.C[i,j,k] + self.C[i-1,j,k]) / self.dx**2 + 
                        (self.C[i,j+1,k] - 2*self.C[i,j,k] + self.C[i,j-1,k]) / self.dy**2
                    )
                    
                    laplacian_v = (self.C[i,j,k+1] - 2*self.C[i,j,k] + self.C[i,j,k-1]) / self.dz**2
                    
                    C_new[i,j,k] += self.dt * (self.Kh * laplacian_h + self.Kv * laplacian_v)
        
        self.C = np.maximum(C_new, 0)
    
    def deposition_step(self):
        """Deposición seca y decaimiento radiactivo"""
        deposition_flux = self.vd * self.C[:, :, 0]
        self.deposition += deposition_flux * self.dt
        self.C[:, :, 0] = np.maximum(self.C[:, :, 0] - (self.vd / self.dz) * self.C[:, :, 0] * self.dt, 0)
        
        self.C *= np.exp(-self.lambda_decay * self.dt)
    
    def apply_boundary_conditions(self):
        """Condiciones de frontera"""
        self.C[0, :, :] = 0
        self.C[-1, :, :] = 0
        self.C[:, 0, :] = 0
        self.C[:, -1, :] = 0
        self.C[:, :, -1] = self.C[:, :, -2]
    
    def step(self):
        """Un paso de tiempo completo"""
        # 🆕 ACTUALIZAR VIENTO PERIÓDICAMENTE
        if (self.time - self.last_wind_update) >= self.wind_update_interval:
            print(f"  🌬️ Actualizando campo de viento en t={self.time_hours:.1f}h")
            self.set_wind_field_time_varying(self.time)
            self.last_wind_update = self.time
        
        self.add_source()
        self.advection_step()
        self.diffusion_step()
        self.deposition_step()
        self.apply_boundary_conditions()
        
        self.time += self.dt
        self.time_hours = self.time / 3600
    
    def get_column_integrated(self):
        """Integración vertical de la concentración"""
        return np.sum(self.C, axis=2) * self.dz
    
    def run_simulation(self, num_steps=240):
        """Ejecutar la simulación"""
        print(f"Iniciando simulación de dispersión de Chernobyl...")
        print(f"Pasos de tiempo: {num_steps}")
        print(f"Tiempo total: {num_steps * self.dt / 3600:.1f} horas")
        
        # Inicializar viento
        self.set_wind_field_time_varying(0)
        
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))
        cfl_x = max_u * self.dt / self.dx
        cfl_y = max_v * self.dt / self.dy
        print(f"CFL x: {cfl_x:.3f}, CFL y: {cfl_y:.3f} (debe ser < 1)")
        
        concentrations = []
        depositions = []
        
        for step in range(num_steps):
            self.step()
            
            if step % 10 == 0:
                concentrations.append(self.get_column_integrated().copy())
                depositions.append(self.deposition.copy())
                print(f"Paso {step}/{num_steps} - Tiempo: {self.time_hours:.1f} h - Max C: {np.max(self.C):.2e}")
        
        return concentrations, depositions


# Resto del código igual...
def visualize_results(model, concentrations, depositions):
    """Visualizar resultados de la simulación"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax1 = axes[0, 0]
    c1 = concentrations[-1]
    extent = [0, model.nx * model.dx / 1000, 0, model.ny * model.dy / 1000]
    im1 = ax1.imshow(c1, origin='lower', cmap='hot', extent=extent,
                     norm=plt.Normalize(vmin=0, vmax=np.percentile(c1, 99)))
    ax1.plot(model.source_x * model.dx / 1000, model.source_y * model.dy / 1000, 
             'c*', markersize=20, label='Chernobyl')
    ax1.set_title('Concentración Integrada en Columna (final)')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='Bq·m/m²')
    
    ax2 = axes[0, 1]
    dep = depositions[-1]
    im2 = ax2.imshow(dep, origin='lower', cmap='YlOrRd', extent=extent,
                     norm=plt.Normalize(vmin=0, vmax=np.percentile(dep, 99)))
    ax2.plot(model.source_x * model.dx / 1000, model.source_y * model.dy / 1000, 
             'b*', markersize=20, label='Chernobyl')
    ax2.set_title('Deposición Acumulada Total')
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='Bq/m²')
    
    ax3 = axes[1, 0]
    vertical_profile = model.C[model.source_x, model.source_y, :]
    heights = np.arange(model.nz) * model.dz / 1000
    ax3.plot(vertical_profile, heights, 'b-', linewidth=2)
    ax3.set_xlabel('Concentración (Bq/m³)')
    ax3.set_ylabel('Altura (km)')
    ax3.set_title('Perfil Vertical en Chernobyl')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    total_mass = [np.sum(c) * model.dx * model.dy for c in concentrations]
    times = np.arange(len(total_mass)) * 10 * model.dt / 3600
    ax4.plot(times, np.array(total_mass) / 1e15, 'g-', linewidth=2)
    ax4.set_xlabel('Tiempo (horas)')
    ax4.set_ylabel('Masa Total en Atmósfera (PBq)')
    ax4.set_title('Evolución de Masa Total Atmosférica')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chernobyl_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)
    print(f"Tiempo simulado: {model.time_hours:.1f} horas ({model.time_hours/24:.1f} días)")
    print(f"Concentración máxima: {np.max(model.C):.2e} Bq/m³")
    print(f"Deposición máxima: {np.max(model.deposition):.2e} Bq/m²")
    print(f"Deposición total: {np.sum(model.deposition) * model.dx * model.dy / 1e15:.2f} PBq")
    print(f"Área afectada (>1e6 Bq/m²): {np.sum(model.deposition > 1e6) * model.dx * model.dy / 1e6:.0f} km²")


if __name__ == "__main__":
    print("="*60)
    print("SIMULACIÓN DE DISPERSIÓN ATMOSFÉRICA - CHERNOBYL 1986")
    print("="*60)
    
    model = ChernobylDispersionModel(
        nx=80, ny=80, nz=15,
        dx=60000, dy=60000, dz=300,
        dt=1200
    )
    
    print("\nEjecutando simulación...")
    concentrations, depositions = model.run_simulation(num_steps=300)
    
    print("\nGenerando visualizaciones...")
    visualize_results(model, concentrations, depositions)