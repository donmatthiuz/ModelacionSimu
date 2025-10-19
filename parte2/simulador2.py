import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

class CellularAutomatonSIR:
    def __init__(self, M, N, I0, r, beta, gamma, T, start_point=None):
        self.M = M
        self.N = N
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.T = T
        
        # Inicializar grid: 0=Susceptible, 1=Infectado, 2=Recuperado
        self.grid = np.zeros((M, N), dtype=int)
        
        # Determinar punto de inicio
        if start_point is None:
            center_m, center_n = M // 2, N // 2
        else:
            center_m, center_n = start_point
        
        infected_count = 0
        
        # Distribuir I0 infectados alrededor del punto de inicio
        for dm in range(-int(np.sqrt(I0)), int(np.sqrt(I0)) + 1):
            for dn in range(-int(np.sqrt(I0)), int(np.sqrt(I0)) + 1):
                if infected_count >= I0:
                    break
                m, n = center_m + dm, center_n + dn
                if 0 <= m < M and 0 <= n < N:
                    self.grid[m, n] = 1
                    infected_count += 1
            if infected_count >= I0:
                break
        
        # Historial
        self.S_hist = []
        self.I_hist = []
        self.R_hist = []
        self.record_counts()
    
    def get_neighborhood(self, i, j):
        """Obtiene las celdas vecinas dentro del radio r"""
        neighbors = []
        for di in range(-self.r, self.r + 1):
            for dj in range(-self.r, self.r + 1):
                if di == 0 and dj == 0:
                    continue
                # Distancia euclidiana
                if np.sqrt(di**2 + dj**2) <= self.r:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.M and 0 <= nj < self.N:
                        neighbors.append((ni, nj))
        return neighbors
    
    def infection_step(self):
        new_grid = self.grid.copy()
        
        for i in range(self.M):
            for j in range(self.N):
                if self.grid[i, j] == 0:  # Susceptible
                    neighbors = self.get_neighborhood(i, j)
                    if len(neighbors) > 0:
                        # Contar vecinos infectados
                        infected_neighbors = sum(
                            1 for ni, nj in neighbors if self.grid[ni, nj] == 1
                        )
                        # Proporción de vecinos infectados
                        infection_proportion = infected_neighbors / len(neighbors)
                        
                        # Probabilidad de infección basada en la proporción
                        if np.random.rand() < self.beta * infection_proportion:
                            new_grid[i, j] = 1
        
        self.grid = new_grid
    
    def recovery_step(self):
        """Actualiza recuperaciones"""
        infected_mask = self.grid == 1
        recovery_mask = np.random.rand(self.M, self.N) < self.gamma
        self.grid[infected_mask & recovery_mask] = 2
    
    def record_counts(self):
        self.S_hist.append(np.sum(self.grid == 0))
        self.I_hist.append(np.sum(self.grid == 1))
        self.R_hist.append(np.sum(self.grid == 2))
    
    def step(self):
        self.infection_step()
        self.recovery_step()
        self.record_counts()

class SIRAutomatonSimulation:
    def __init__(self, M=100, N=100, I0=10, r=2, beta=0.3, gamma=0.05, T=200, start_point=None):
        self.ca = CellularAutomatonSIR(M, N, I0, r, beta, gamma, T, start_point)
        self.T = T
        self.M = M
        self.N = N
    
    def run(self, save_animation=True):
        colors = ['#00FF00', '#FF0000', '#0000FF']
        cmap = ListedColormap(colors)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel izquierdo: grid del autómata
        im = ax1.imshow(self.ca.grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        ax1.set_title('Autómata Celular SIR')
        ax1.axis('off')
        
        # Barra de color
        cbar = plt.colorbar(im, ax=ax1, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.set_ticklabels(['Susceptible', 'Infectado', 'Recuperado'])
        
        # Panel derecho: curvas SIR
        line_s, = ax2.plot([], [], 'g-', label='S', linewidth=2)
        line_i, = ax2.plot([], [], 'r-', label='I', linewidth=2)
        line_r, = ax2.plot([], [], 'b-', label='R', linewidth=2)
        ax2.set_xlim(0, self.T)
        ax2.set_ylim(0, self.M * self.N)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Número de celdas')
        ax2.set_title('Evolución SIR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        def update(frame):
            if frame > 0:
                self.ca.step()
            
            # Actualizar grid
            im.set_array(self.ca.grid)
            S_count = np.sum(self.ca.grid == 0)
            I_count = np.sum(self.ca.grid == 1)
            R_count = np.sum(self.ca.grid == 2)
            ax1.set_title(f't = {frame} | S = {S_count} | I = {I_count} | R = {R_count}')
            
            # Actualizar curvas
            t = np.arange(len(self.ca.S_hist))
            line_s.set_data(t, self.ca.S_hist)
            line_i.set_data(t, self.ca.I_hist)
            line_r.set_data(t, self.ca.R_hist)
            
            return im, line_s, line_i, line_r
        
        ani = FuncAnimation(fig, update, frames=self.T, interval=50, blit=True)
        
        if save_animation:
            print("Guardando animación...")
            ani.save("sir_automaton.gif", writer=PillowWriter(fps=20))
            print("Animación guardada como 'sir_automaton.gif'")
        
        plt.tight_layout()
        plt.show()
    
    def plot_curves(self, save_figure=True):
        t = np.arange(len(self.ca.S_hist))
        plt.figure(figsize=(10, 6))
        plt.plot(t, self.ca.S_hist, 'g-', label="S (Susceptibles)", linewidth=2)
        plt.plot(t, self.ca.I_hist, 'r-', label="I (Infectados)", linewidth=2)
        plt.plot(t, self.ca.R_hist, 'b-', label="R (Recuperados)", linewidth=2)
        plt.xlabel("Tiempo")
        plt.ylabel("Número de celdas")
        plt.title("Evolución del modelo SIR con Autómata Celular")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_figure:
            plt.savefig("sir_automaton_curves.png", dpi=300, bbox_inches='tight')
            print("Gráfica guardada como 'sir_automaton_curves.png'")
        
        plt.show()
    
    def plot_final_state(self, save_figure=True):
        """Grafica el estado final del autómata"""
        colors = ['#00FF00', '#FF0000', '#0000FF']
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(10, 10))
        im = plt.imshow(self.ca.grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        
        S_count = np.sum(self.ca.grid == 0)
        I_count = np.sum(self.ca.grid == 1)
        R_count = np.sum(self.ca.grid == 2)
        plt.title(f"Estado Final | S = {S_count} | I = {I_count} | R = {R_count}")
        plt.axis('off')
        
        cbar = plt.colorbar(im, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.set_ticklabels(['Susceptible', 'Infectado', 'Recuperado'])
        
        if save_figure:
            plt.savefig("sir_automaton_final.png", dpi=300, bbox_inches='tight')
            print("Estado final guardado como 'sir_automaton_final.png'")
        
        plt.show()

if __name__ == "__main__":
    # Parámetros de la simulación
    sim = SIRAutomatonSimulation(
        M=100,          # Altura del grid
        N=100,          # Anchura del grid
        I0=20,          # Número inicial de infectados
        r=2,            # Radio de vecindad
        beta=0.4,       # Tasa de infección
        gamma=0.05,     # Tasa de recuperación
        T=250,          # Tiempo total de simulación
        start_point=(20, 20)  # Punto de inicio (fila, columna)
                              # None para centro
                              # (0, 0) para esquina superior izquierda
                              # (99, 99) para esquina inferior derecha
    )
    
    # Ejecutar simulación y guardar animación
    sim.run(save_animation=True)
    
    # Guardar gráficas finales
    sim.plot_curves(save_figure=True)
    sim.plot_final_state(save_figure=True)
    
    print("\n¡Simulación completada! Se generaron:")
    print("- sir_automaton.gif")
    print("- sir_automaton_curves.png")
    print("- sir_automaton_final.png")  