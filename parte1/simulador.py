import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class ParticleSystem:
    def __init__(self, L, N, I0, vmax, r, beta, gamma, dt):
        self.L = L
        self.N = N
        self.vmax = vmax
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.pos = np.random.rand(N, 2) * L
        self.vel = (np.random.rand(N, 2) - 0.5) * 2 * vmax
        self.state = np.zeros(N, dtype=int)
        infected_idx = np.random.choice(N, I0, replace=False)
        self.state[infected_idx] = 1
        self.S_hist, self.I_hist, self.R_hist = [], [], []

    def update_positions(self):
        self.pos += self.vel * self.dt
        for i in range(2):
            low = self.pos[:, i] < 0
            high = self.pos[:, i] > self.L
            self.vel[low | high, i] *= -1
            self.pos[low, i] = -self.pos[low, i]
            self.pos[high, i] = 2 * self.L - self.pos[high, i]

    def infection_step(self):
        # Encontrar todas las partículas infectadas
        infected = np.where(self.state == 1)[0]
        susceptible = np.where(self.state == 0)[0]
        
        if len(infected) == 0 or len(susceptible) == 0:
            return
        
        for s_idx in susceptible:
            distances = np.linalg.norm(self.pos[infected] - self.pos[s_idx], axis=1)
            # Si alguno está dentro del radio de infección
            if np.any(distances < self.r):
                # Probabilidad de infección
                if np.random.rand() < self.beta * self.dt:
                    self.state[s_idx] = 1

    def recovery_step(self):
        infected = self.state == 1
        recovery_prob = np.random.rand(self.N) < self.gamma * self.dt
        self.state[infected & recovery_prob] = 2

    def record_counts(self):
        self.S_hist.append(np.sum(self.state == 0))
        self.I_hist.append(np.sum(self.state == 1))
        self.R_hist.append(np.sum(self.state == 2))

    def step(self):
        self.update_positions()
        self.infection_step()
        self.recovery_step()
        self.record_counts()

class SIRSimulation:
    def __init__(self, L=10.0, N=200, I0=5, vmax=0.1, r=0.2, beta=0.8, gamma=0.1, dt=0.1, steps=400):
        self.sys = ParticleSystem(L, N, I0, vmax, r, beta, gamma, dt)
        self.steps = steps
        self.L = L
        self.dt = dt

    def run(self, save_animation=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel izquierdo: partículas
        sc = ax1.scatter(self.sys.pos[:, 0], self.sys.pos[:, 1], 
                        c=self.sys.state, cmap='RdYlGn_r', vmin=0, vmax=2, s=50)
        ax1.set_xlim(0, self.L)
        ax1.set_ylim(0, self.L)
        ax1.set_aspect('equal')
        
        # Panel derecho: curvas SIR
        line_s, = ax2.plot([], [], 'g-', label='S', linewidth=2)
        line_i, = ax2.plot([], [], 'y-', label='I', linewidth=2)
        line_r, = ax2.plot([], [], 'r-', label='R', linewidth=2)
        ax2.set_xlim(0, self.steps * self.dt)
        ax2.set_ylim(0, self.sys.N)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Número de individuos')
        ax2.set_title('Evolución SIR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        def update(frame):
            self.sys.step()
            
            # Actualizar partículas
            sc.set_offsets(self.sys.pos)
            sc.set_array(self.sys.state)
            S_count = np.sum(self.sys.state == 0)
            I_count = np.sum(self.sys.state == 1)
            R_count = np.sum(self.sys.state == 2)
            ax1.set_title(f"t = {frame*self.dt:.1f} | S = {S_count} | I = {I_count} | R = {R_count}")
            
            # Actualizar curvas
            t = np.arange(len(self.sys.S_hist)) * self.dt
            line_s.set_data(t, self.sys.S_hist)
            line_i.set_data(t, self.sys.I_hist)
            line_r.set_data(t, self.sys.R_hist)
            
            return sc, line_s, line_i, line_r

        ani = FuncAnimation(fig, update, frames=self.steps, interval=50, blit=True)
        
        if save_animation:
            print("Guardando animación...")
            ani.save("sir_particles.gif", writer=PillowWriter(fps=20))
            print("Animación guardada como 'sir_particles.gif'")
        
        plt.tight_layout()
        plt.show()

    def plot_curves(self, save_figure=True):
        t = np.arange(len(self.sys.S_hist)) * self.dt
        plt.figure(figsize=(8, 6))
        plt.plot(t, self.sys.S_hist, 'g-', label="S (Susceptibles)", linewidth=2)
        plt.plot(t, self.sys.I_hist, 'y-', label="I (Infectados)", linewidth=2)
        plt.plot(t, self.sys.R_hist, 'r-', label="R (Recuperados)", linewidth=2)
        plt.xlabel("Tiempo")
        plt.ylabel("Número de individuos")
        plt.title("Evolución del modelo SIR")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_figure:
            plt.savefig("sir_curves.png", dpi=300, bbox_inches='tight')
            print("Gráfica guardada como 'sir_curves.png'")
        
        plt.show()

    def plot_final_state(self, save_figure=True):
        """Guarda el estado final de las partículas"""
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(self.sys.pos[:, 0], self.sys.pos[:, 1], 
                        c=self.sys.state, cmap='RdYlGn_r', vmin=0, vmax=2, s=100, alpha=0.6)
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        
        S_count = np.sum(self.sys.state == 0)
        I_count = np.sum(self.sys.state == 1)
        R_count = np.sum(self.sys.state == 2)
        plt.title(f"Estado Final | S = {S_count} | I = {I_count} | R = {R_count}")
        
        cbar = plt.colorbar(sc, ticks=[0, 1, 2])
        cbar.set_ticklabels(['Susceptible', 'Infectado', 'Recuperado'])
        
        if save_figure:
            plt.savefig("sir_final_state.png", dpi=300, bbox_inches='tight')
            print("Estado final guardado como 'sir_final_state.png'")
        
        plt.show()

if __name__ == "__main__":
    sim = SIRSimulation(
        L=10.0,
        N=200,
        I0=5,
        vmax=0.8,
        r=0.3,      
        beta=1.5,  
        gamma=0.05,
        dt=0.1,
        steps=250
    )
    
    # Ejecutar simulación y guardar GIF
    sim.run(save_animation=True)
    
    # Guardar gráficas finales
    sim.plot_curves(save_figure=True)
    sim.plot_final_state(save_figure=True)
    
    print("\n¡Simulación completada! Se generaron:")
    print("- sir_particles.gif")
    print("- sir_curves.png")
    print("- sir_final_state.png")