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
KM_PER_DEG_LAT = 111  # Approx
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

    model = ChernobylDispersionModel(nx=nx, ny=ny, nz=15, dx=dx, dy=dy, dz=300, dt=1800)
    model.set_wind_field(u_mean=6.0, v_mean=4.0, variability=2.5)
    concentrations, depositions = model.run_simulation(num_steps=300)

    sim_window = tk.Toplevel()
    sim_window.title("Simulación atmosférica")
    sim_window.geometry("1000x900")

    fig = Figure(figsize=(7.8, 7.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#e5ddc8")
    ax.add_feature(cfeature.OCEAN, facecolor="#9cc9ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    # initial frame
    first_frame = concentrations[0].T
    im = ax.imshow(first_frame, origin='lower', cmap='hot',
                   extent=[lon_min, lon_max, lat_min, lat_max],
                   norm=plt.Normalize(vmin=0, vmax=np.percentile(first_frame, 99)),
                   transform=ccrs.PlateCarree(), alpha=0.8)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04)
    cbar.set_label("Concentración integrada")

    canvas = FigureCanvasTkAgg(fig, sim_window)
    canvas.get_tk_widget().pack(expand=True)
    canvas.draw()

    # ---- ANIMATION LOOP ----
    def animate(i=0):
        if i < len(concentrations):
            frame = concentrations[i].T
            im.set_data(frame)
            im.set_norm(plt.Normalize(vmin=0, vmax=np.percentile(frame, 99)))
            canvas.draw()
            sim_window.after(150, lambda: animate(i+1))  # 150ms delay

    animate()  # start animation


# -----------------------------------------
# Ventana principal original
# -----------------------------------------

window = tk.Tk()
window.title("Chernobyl simulation")
window.geometry("700x700")

container = tk.Frame(window)
container.pack(expand=True)

tk.Label(container, text="Cantidad de puntos en X (nx): ").pack()
entry1 = tk.Entry(container)
entry1.pack(pady=5)

tk.Label(container, text="Cantidad de puntos en Y (ny): ").pack()
entry2 = tk.Entry(container)
entry2.pack(pady=5)

tk.Label(container, text="Tamaño de cada celda X (m): ").pack()
entry3 = tk.Entry(container)
entry3.pack(pady=5)

tk.Label(container, text="Tamaño de cada celda Y (m): ").pack()
entry4 = tk.Entry(container)
entry4.pack(pady=5)

tk.Button(
    container,
    text="Abrir Mapa",
    command=lambda: open_simulation_window(entry1.get(), entry2.get(), entry3.get(), entry4.get())
).pack(pady=15)

window.mainloop()
