import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from DispersionModel import ChernobylDispersionModel

CHERNOBYL_LON = 30.1
CHERNOBYL_LAT = 51.4
KM_PER_DEG_LAT = 111

def km_to_deg_lon(km, lat_deg):
    return km / (KM_PER_DEG_LAT * np.cos(np.radians(lat_deg)))

def open_simulation_window(nx_str, ny_str, dx_str, dy_str):
    try:
        nx = int(nx_str)
        ny = int(ny_str)
        dx = float(dx_str)
        dy = float(dy_str)
    except ValueError:
        messagebox.showerror("Error", "nx, ny, dx y dy deben ser numéricos.")
        return
    
    total_km_x = (nx * dx) / 1000.0
    total_km_y = (ny * dy) / 1000.0
    
    half_width_deg  = km_to_deg_lon(total_km_x / 2, CHERNOBYL_LAT)
    half_height_deg = (total_km_y / 2) / KM_PER_DEG_LAT
    
    lon_min = CHERNOBYL_LON - half_width_deg
    lon_max = CHERNOBYL_LON + half_width_deg
    lat_min = CHERNOBYL_LAT - half_height_deg
    lat_max = CHERNOBYL_LAT + half_height_deg

    print("Creando modelo...")
    model = ChernobylDispersionModel(nx=nx, ny=ny, nz=15, dx=dx, dy=dy, dz=300, dt=1200)
    
    # 🔧 ELIMINADO: Ya no llamamos set_wind_field aquí
    # El viento se inicializa automáticamente en run_simulation()
    
    print("Ejecutando simulación...")
    concentrations, depositions = model.run_simulation(num_steps=300)
    
    print("Abriendo ventana de visualización...")
    sim_window = tk.Toplevel()
    sim_window.title("Simulación atmosférica - Chernobyl 1986")  # 🎨 Título mejorado
    sim_window.geometry("1000x900")

    fig = Figure(figsize=(7.8, 7.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#e5ddc8")
    ax.add_feature(cfeature.OCEAN, facecolor="#9cc9ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
    ax.add_feature(cfeature.LAKES, facecolor="#9cc9ff", alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    first_frame = concentrations[0]
    
    all_data = np.concatenate([c.flatten() for c in concentrations])
    vmax = np.percentile(all_data[all_data > 0], 99) if np.any(all_data > 0) else 1
    
    im = ax.imshow(first_frame, origin='lower', cmap='hot',
                   extent=[lon_min, lon_max, lat_min, lat_max],
                   norm=plt.Normalize(vmin=0, vmax=vmax),
                   transform=ccrs.PlateCarree(), alpha=0.7,
                   interpolation='bilinear')

    ax.plot(CHERNOBYL_LON, CHERNOBYL_LAT, 'c*', markersize=20, 
            transform=ccrs.PlateCarree(), label='Chernobyl', zorder=10)
    ax.legend(loc='upper right')

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04)
    cbar.set_label("Concentración integrada (Bq·m/m²)")

    title_text = ax.text(0.5, 1.02, 'Tiempo: 0.0 horas', 
                        transform=ax.transAxes, ha='center', fontsize=12, weight='bold')

    canvas = FigureCanvasTkAgg(fig, sim_window)
    canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
    canvas.draw()

    control_frame = tk.Frame(sim_window)
    control_frame.pack(pady=5)
    
    is_playing = [True]
    current_frame = [0]
    
    def toggle_play():
        is_playing[0] = not is_playing[0]
        play_button.config(text="⏸ Pausar" if is_playing[0] else "▶ Reproducir")
    
    def restart():
        current_frame[0] = 0
        is_playing[0] = True
        play_button.config(text="⏸ Pausar")
    
    play_button = tk.Button(control_frame, text="⏸ Pausar", command=toggle_play, width=15)
    play_button.pack(side=tk.LEFT, padx=5)
    
    restart_button = tk.Button(control_frame, text="⏮ Reiniciar", command=restart, width=15)
    restart_button.pack(side=tk.LEFT, padx=5)

    def animate():
        if is_playing[0] and current_frame[0] < len(concentrations):
            frame = concentrations[current_frame[0]]
            im.set_data(frame)
            
            time_hours = current_frame[0] * 10 * model.dt / 3600
            title_text.set_text(f'Tiempo: {time_hours:.1f} horas ({time_hours/24:.1f} días)')
            
            canvas.draw()
            current_frame[0] += 1
        
        if current_frame[0] < len(concentrations):
            sim_window.after(150, animate)
        else:
            is_playing[0] = False
            play_button.config(text="▶ Reproducir")

    animate()


# -----------------------------------------
# Ventana principal
# -----------------------------------------

window = tk.Tk()
window.title("Chernobyl Dispersion Simulator")  # 🎨 Título profesional
window.geometry("700x700")

container = tk.Frame(window)
container.pack(expand=True)

tk.Label(container, text="Cantidad de puntos en X (nx): ", font=("Arial", 12)).pack(pady=5)
entry1 = tk.Entry(container, font=("Arial", 11))
entry1.insert(0, "80")  # ✅ Mantener 80
entry1.pack(pady=5)

tk.Label(container, text="Cantidad de puntos en Y (ny): ", font=("Arial", 12)).pack(pady=5)
entry2 = tk.Entry(container, font=("Arial", 11))
entry2.insert(0, "80")  # ✅ Mantener 80
entry2.pack(pady=5)

tk.Label(container, text="Tamaño de cada celda X (m): ", font=("Arial", 12)).pack(pady=5)
entry3 = tk.Entry(container, font=("Arial", 11))
entry3.insert(0, "50000")  # 🔧 CAMBIO: 50km en vez de 60km
entry3.pack(pady=5)

tk.Label(container, text="Tamaño de cada celda Y (m): ", font=("Arial", 12)).pack(pady=5)
entry4 = tk.Entry(container, font=("Arial", 11))
entry4.insert(0, "50000")  # 🔧 CAMBIO: 50km en vez de 60km
entry4.pack(pady=5)

# Información adicional
info_frame = tk.Frame(container, bg="#f0f0f0", relief=tk.RIDGE, borderwidth=2)
info_frame.pack(pady=20, padx=20, fill=tk.X)

tk.Label(info_frame, text="ℹ️ Configuraciones recomendadas:", 
         font=("Arial", 10, "bold"), bg="#f0f0f0").pack(pady=5)

# 🔧 ACTUALIZADO: Mejores recomendaciones
config_text = """
Rápida:  nx=50,  ny=50,  dx=80000, dy=80000  (~200×200 km²)
Media:   nx=80,  ny=80,  dx=50000, dy=50000  (~200×200 km²) ⭐
Alta:    nx=120, ny=120, dx=35000, dy=35000  (~210×210 km²)
Extrema: nx=150, ny=150, dx=30000, dy=30000  (~225×225 km²)
"""
tk.Label(info_frame, text=config_text, font=("Courier", 9), 
         bg="#f0f0f0", justify=tk.LEFT).pack(pady=5)

# 🆕 AGREGAR: Nota sobre el tiempo de simulación
note_frame = tk.Frame(container, bg="#fff3cd", relief=tk.RIDGE, borderwidth=2)
note_frame.pack(pady=10, padx=20, fill=tk.X)

tk.Label(note_frame, text="⏱️ Tiempo estimado de ejecución:", 
         font=("Arial", 9, "bold"), bg="#fff3cd").pack(pady=3)

time_text = """
Rápida: ~30 segundos | Media: ~2 minutos | Alta: ~5 minutos
"""
tk.Label(note_frame, text=time_text, font=("Arial", 8), 
         bg="#fff3cd", justify=tk.CENTER).pack(pady=3)

tk.Button(
    container,
    text="🚀 Iniciar Simulación",
    command=lambda: open_simulation_window(entry1.get(), entry2.get(), entry3.get(), entry4.get()),
    font=("Arial", 14, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10
).pack(pady=20)

window.mainloop()