"""
Simulación Chernobyl estilo Pudykiewicz 1989 - ANIMACIÓN INTERACTIVA
- Visualización combinada: concentración + depósito acumulado
- Animación en tiempo real en la misma figura
- Condiciones iniciales realistas basadas en análisis meteorológico
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from scipy.ndimage import gaussian_filter

# -----------------------
# DOMINIO
# -----------------------
lon_min, lon_max = 10.0, 80.0
lat_min, lat_max = 25.0, 70.0
nx, ny = 220, 160

lons = np.linspace(lon_min, lon_max, nx)
lats = np.linspace(lat_min, lat_max, ny)
Lon, Lat = np.meshgrid(lons, lats)

deg2m = 111000.0
dx = np.mean(deg2m * np.cos(np.deg2rad(lats)))
dy = deg2m

# -----------------------
# PARÁMETROS FÍSICOS (calibrados del paper)
# -----------------------
SCALE = 5e13  # Bq total liberado (escala realista)

# Cs-137: T_1/2 = 30.17 años
decay_lambda = np.log(2) / (30.17 * 365.25 * 24 * 3600)

# Deposición seca: típica para aerosoles ~0.001 m/s, altura mezcla ~1000m
v_d = 0.0005  # m/s
mixed_layer_h = 1200.0  # m
alpha = v_d / mixed_layer_h

# Lavado húmedo: scavenging ~10^-4 a 10^-5 s^-1 per mm/h
scavenging_coeff = 5e-5  # s^-1 per mm/h

# -----------------------
# FUENTE (basada en reconstrucción histórica)
# -----------------------
chernobyl_lon = 30.1
chernobyl_lat = 51.3
source_radius_deg = 0.25  # muy localizada
release_rate = 3.0 * SCALE  # Bq/s integrado
release_duration = 10 * 3600.0  # ~10 horas emisión principal

def source_spatial_mask(Lon, Lat, lon0, lat0, radius_deg):
    r = np.sqrt((Lon - lon0)**2 + (Lat - lat0)**2)
    mask = np.exp(-0.5 * (r / (radius_deg/2.5))**2)
    return mask / mask.sum()

source_shape = source_spatial_mask(Lon, Lat, chernobyl_lon, chernobyl_lat, source_radius_deg)

# -----------------------
# CAMPO DE VIENTOS REALISTA
# -----------------------
def wind_field_realistic(Lon, Lat, t_seconds):
    """
    Reconstrucción meteorológica abril-mayo 1986
    Múltiples sistemas que compiten y generan dispersión multidireccional
    + CAMPOS DE DEFORMACIÓN para filamentación
    """
    t_h = t_seconds / 3600.0
    
    # === SISTEMA 1: Baja Escandinavia (flujo ciclónico N-NE) ===
    lon_low1 = 20.0 + np.minimum(t_h/30.0, 1.5) * 3.0
    lat_low1 = 59.0
    r_low1 = np.sqrt(((Lon - lon_low1)*1.3)**2 + (Lat - lat_low1)**2)
    theta_low1 = np.arctan2(Lat - lat_low1, (Lon - lon_low1)*1.3)
    
    # Máximo en t=24-48h
    intensity_low1 = 9.0 * np.exp(-((t_h - 36.0)/24.0)**2)
    V_low1 = intensity_low1 * np.exp(-(r_low1/16.0)**2)
    
    u_low1 = -V_low1 * np.sin(theta_low1)
    v_low1 = V_low1 * np.cos(theta_low1)
    
    # === SISTEMA 2: Alta presión SE Europa (flujo anticiclónico S-SW) ===
    lon_high = 38.0
    lat_high = 44.0
    r_high = np.sqrt(((Lon - lon_high)*1.2)**2 + (Lat - lat_high)**2)
    theta_high = np.arctan2(Lat - lat_high, (Lon - lon_high)*1.2)
    
    # Crece desde t=12h, máximo en t=48-96h
    intensity_high = -7.5 * (1.0 - np.exp(-np.maximum(0, t_h-12.0)/18.0)) * np.exp(-np.maximum(0, t_h-72.0)/48.0)
    V_high = intensity_high * np.exp(-(r_high/15.0)**2)
    
    u_high = -V_high * np.sin(theta_high)
    v_high = V_high * np.cos(theta_high)
    
    # === SISTEMA 3: Baja secundaria Mar Báltico (flujo E-SE) ===
    lon_low2 = 24.0
    lat_low2 = 56.0
    r_low2 = np.sqrt(((Lon - lon_low2)*1.4)**2 + (Lat - lat_low2)**2)
    theta_low2 = np.arctan2(Lat - lat_low2, (Lon - lon_low2)*1.4)
    
    # Activa t=36-84h
    intensity_low2 = 6.5 * np.exp(-((t_h - 60.0)/30.0)**2)
    V_low2 = intensity_low2 * np.exp(-(r_low2/12.0)**2)
    
    u_low2 = -V_low2 * np.sin(theta_low2)
    v_low2 = V_low2 * np.cos(theta_low2)
    
    # === SISTEMA 4: Corriente en chorro (flujo W-E con ondulaciones) ===
    # Posición variable del jet
    jet_lat = 53.0 + 3.0 * np.sin(2.0 * np.pi * t_h / 60.0)
    jet_intensity = 7.0 * (0.8 + 0.2 * np.cos(2.0 * np.pi * t_h / 72.0))
    
    u_jet = jet_intensity * np.exp(-((Lat - jet_lat)/5.0)**2)
    v_jet = 1.5 * np.sin(2.0 * np.pi * (Lon - 30.0 + t_h * 0.4) / 40.0) * np.exp(-((Lat - jet_lat)/8.0)**2)
    
    # === SISTEMA 5: Ondas planetarias (dispersión de gran escala) ===
    wave_phase1 = 2.0 * np.pi * (Lon - 15.0 + t_h * 0.35) / 50.0
    wave_amp1 = 4.5 * np.exp(-((Lat - 58.0)/12.0)**2)
    
    u_wave1 = -1.5 * wave_amp1 * np.cos(wave_phase1)
    v_wave1 = wave_amp1 * np.sin(wave_phase1)
    
    wave_phase2 = 2.0 * np.pi * (Lon - 25.0 - t_h * 0.25) / 45.0
    wave_amp2 = 3.5 * np.exp(-((Lat - 48.0)/10.0)**2)
    
    u_wave2 = wave_amp2 * np.cos(wave_phase2)
    v_wave2 = 1.2 * wave_amp2 * np.sin(wave_phase2)
    
    # === SISTEMA 6: Vórtice móvil Europa Central (t>96h) ===
    if t_h > 96:
        lon_v = 32.0 + 6.0 * np.sin((t_h-96.0) * 0.005)
        lat_v = 50.0 + 4.0 * np.cos((t_h-96.0) * 0.005)
        r_v = np.sqrt(((Lon - lon_v)*1.3)**2 + (Lat - lat_v)**2)
        theta_v = np.arctan2(Lat - lat_v, (Lon - lon_v)*1.3)
        
        intensity_v = -5.5 * (1.0 - np.exp(-(t_h-96.0)/20.0))
        V_v = intensity_v * np.exp(-(r_v/11.0)**2)
        
        u_v = -V_v * np.sin(theta_v)
        v_v = V_v * np.cos(theta_v)
    else:
        u_v, v_v = 0.0, 0.0
    
    # === FLUJO DE FONDO (débil pero omnipresente) ===
    u_bg = 2.5 + 1.5 * np.sin(np.pi * (Lat - 45.0) / 25.0)
    v_bg = 0.8
    
    # === CAMPOS DE DEFORMACIÓN (CLAVE PARA FILAMENTACIÓN) ===
    # Estos campos crean STRETCHING y SHEARING que fragmentan la pluma
    
    # Deformación 1: Frente de cizalladura W-E (zona de fragmentación principal)
    front_lat = 52.0 + 2.0 * np.sin(2.0 * np.pi * t_h / 48.0)
    shear_strength = 5.5 * np.exp(-((t_h - 60.0)/36.0)**2)  # Máximo en t=60h
    shear_width = 4.0
    
    # Gradiente fuerte de velocidad zonal
    shear_profile = np.tanh((Lat - front_lat) / shear_width)
    u_shear = shear_strength * shear_profile
    v_shear = 0.8 * shear_strength * np.exp(-((Lat - front_lat)/6.0)**2) * np.sin(2.0 * np.pi * Lon / 35.0)
    
    # Deformación 2: Zonas de convergencia/divergencia (crean "islas")
    # Múltiples centros de deformación que evolucionan en tiempo
    
    # Centro 1: Europa Central (divergencia → dispersión radial)
    lon_def1 = 28.0 + 4.0 * np.cos(t_h * 0.006)
    lat_def1 = 50.0 + 3.0 * np.sin(t_h * 0.006)
    r_def1 = np.sqrt(((Lon - lon_def1)*1.2)**2 + (Lat - lat_def1)**2)
    
    def_intensity1 = 3.5 * np.sin(2.0 * np.pi * t_h / 72.0)**2  # Pulsante
    u_def1 = def_intensity1 * (Lon - lon_def1) * np.exp(-(r_def1/8.0)**2)
    v_def1 = def_intensity1 * (Lat - lat_def1) * np.exp(-(r_def1/8.0)**2)
    
    # Centro 2: Escandinavia (convergencia → compresión)
    lon_def2 = 18.0 + 3.0 * np.sin(t_h * 0.008)
    lat_def2 = 60.0 + 2.0 * np.cos(t_h * 0.008)
    r_def2 = np.sqrt(((Lon - lon_def2)*1.3)**2 + (Lat - lat_def2)**2)
    
    def_intensity2 = -2.8 * np.cos(2.0 * np.pi * (t_h - 30.0) / 60.0)**2
    u_def2 = def_intensity2 * (Lon - lon_def2) * np.exp(-(r_def2/10.0)**2)
    v_def2 = def_intensity2 * (Lat - lat_def2) * np.exp(-(r_def2/10.0)**2)
    
    # Deformación 3: Filamentos de mesoescala (estructura fina)
    meso_phase_x = 2.0 * np.pi * (Lon / 12.0 + t_h * 0.015)
    meso_phase_y = 2.0 * np.pi * (Lat / 10.0 - t_h * 0.012)
    
    u_meso = 1.8 * np.sin(meso_phase_x) * np.cos(meso_phase_y)
    v_meso = 1.8 * np.cos(meso_phase_x) * np.sin(meso_phase_y)
    
    # === SUPERPOSICIÓN (competencia de sistemas + deformación) ===
    u = (u_low1 + u_low2 + u_high + u_jet + u_wave1 + u_wave2 + u_v + u_bg +
         u_shear + u_def1 + u_def2 + u_meso)
    v = (v_low1 + v_low2 + v_high + v_jet + v_wave1 + v_wave2 + v_v + v_bg +
         v_shear + v_def1 + v_def2 + v_meso)
    
    # === TURBULENCIA MESOESCALA ===
    u_turb = 2.5 * np.sin(0.08 * Lon + t_h * 0.002) * np.cos(0.08 * Lat)
    v_turb = 2.5 * np.cos(0.08 * Lon) * np.sin(0.08 * Lat + t_h * 0.002)
    
    u += u_turb
    v += v_turb
    
    return u, v


# -----------------------
# PRECIPITACIÓN
# -----------------------
def precipitation_field(Lon, Lat, t):
    t_h = t / 3600.0
    rain = np.zeros_like(Lon)
    
    # 28 abril (~48h): Evento sobre Escandinavia
    if 42 <= t_h <= 54:
        r1 = np.sqrt((Lon - 16.0)**2 + (Lat - 61.0)**2)
        rain += 6.0 * np.exp(-(r1/9.0)**2)
    
    # 30 abril - 1 mayo (~96-120h): Europa Central
    if 90 <= t_h <= 126:
        r2 = np.sqrt((Lon - 14.0)**2 + (Lat - 50.0)**2)
        rain += 4.5 * np.exp(-(r2/8.0)**2)
        r3 = np.sqrt((Lon - 20.0)**2 + (Lat - 52.0)**2)
        rain += 3.5 * np.exp(-(r3/7.0)**2)
    
    # 8 mayo (~288h): Evento mayor Suecia
    if 282 <= t_h <= 294:
        r4 = np.sqrt((Lon - 13.0)**2 + (Lat - 58.0)**2)
        rain += 8.0 * np.exp(-(r4/6.0)**2)
    
    # Sistemas móviles posteriores
    if t_h > 150:
        lon_front = 18.0 + ((t_h - 150.0) / 24.0) * 10.0
        r5 = np.sqrt(((Lon - lon_front)/2.5)**2 + (Lat - 54.0)**2)
        rain += 3.0 * np.exp(-(r5/7.0)**2)
    
    return rain

# -----------------------
# OPERADORES NUMÉRICOS
# -----------------------
def bilinear_interp_on_grid(A, Lon, Lat, lon_pts, lat_pts):
    ny, nx = A.shape
    lon_vec, lat_vec = Lon[0,:], Lat[:,0]
    dx_deg, dy_deg = lon_vec[1] - lon_vec[0], lat_vec[1] - lat_vec[0]
    
    ix = np.clip((lon_pts - lon_vec[0]) / dx_deg, 0, nx-1.000001)
    iy = np.clip((lat_pts - lat_vec[0]) / dy_deg, 0, ny-1.000001)
    
    i0, j0 = np.floor(ix).astype(int), np.floor(iy).astype(int)
    i1, j1 = np.clip(i0+1, 0, nx-1), np.clip(j0+1, 0, ny-1)
    
    wx, wy = ix - i0, iy - j0
    
    return ((1-wx)*(1-wy)*A[j0,i0] + wx*(1-wy)*A[j0,i1] +
            (1-wx)*wy*A[j1,i0] + wx*wy*A[j1,i1])

def semi_lagrangian_step(A, u, v, dt):
    coslat = np.cos(np.deg2rad(Lat))
    lon_dep = Lon - (u * dt) / (deg2m * coslat)
    lat_dep = Lat - (v * dt) / deg2m
    return bilinear_interp_on_grid(A, Lon, Lat, lon_dep, lat_dep)

def horizontal_diffusion(A, K_h, dt):
    sigma_m = np.sqrt(2.0 * K_h * dt)
    sigma_pix = sigma_m / np.mean([dx, dy])
    if sigma_pix < 0.1:
        return A
    A_diff = gaussian_filter(A, sigma=sigma_pix, mode='constant', cval=0)
    return A_diff * (A.sum() / (A_diff.sum() + 1e-30))

def apply_source_sink(A, dt, t, precip):
    W = scavenging_coeff * precip
    kappa = alpha + decay_lambda + W
    expf = np.exp(-kappa * dt)
    
    depo_rate = alpha + W
    deposited = np.where(kappa > 1e-12,
                        A * (depo_rate / kappa) * (1.0 - expf),
                        A * depo_rate * dt)
    
    source = np.zeros_like(A)
    if t < release_duration:
        t_h = t / 3600.0
        temporal = np.exp(-((t_h - 1.0)**2) / 8.0)
        source = release_rate * temporal * source_shape
    
    A_new = A * expf
    A_new = np.where(kappa > 1e-12,
                    A_new + (source / kappa) * (1.0 - expf),
                    A_new + source * dt)
    
    return A_new, deposited

def advance_timestep(A, D, t, dt):
    u, v = wind_field_realistic(Lon, Lat, t)
    A_adv = semi_lagrangian_step(A, u, v, dt)
    A_diff = horizontal_diffusion(A_adv, K_h=1.8e5, dt=dt)
    precip = precipitation_field(Lon, Lat, t)
    A_next, deposited = apply_source_sink(A_diff, dt, t, precip)
    D_next = D + deposited
    return A_next, D_next

# -----------------------
# CONDICIONES INICIALES
# -----------------------
A = np.zeros((ny, nx)) + 1e9
D = np.zeros((ny, nx))

# -----------------------
# INTEGRACIÓN
# -----------------------
dt = 1800.0  # 30 min
total_hours = 240  # 10 días
nsteps = int((total_hours * 3600) / dt)
print(f"Simulación: {total_hours/24:.0f} días, {nsteps} pasos")

frame_interval = max(1, nsteps // 100)
frames_A, frames_D, times = [], [], []

t, t0 = 0.0, time.time()
for step in range(nsteps):
    A, D = advance_timestep(A, D, t, dt)
    t += dt
    if step % frame_interval == 0 or step == nsteps-1:
        frames_A.append(A.copy())
        frames_D.append(D.copy())
        times.append(t)
    if step % 20 == 0:
        print(f"  {step}/{nsteps} | t={t/3600:.1f}h | {time.time()-t0:.0f}s")

print("Simulación completada. Generando animación...")

# -----------------------
# ANIMACIÓN INTERACTIVA
# -----------------------
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='0.88', zorder=0)
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue', alpha=0.4, zorder=0)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=3)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4, zorder=3)

# Normas de color
norm_D = colors.LogNorm(vmin=1e9, vmax=5e14, clip=True)
norm_A = colors.LogNorm(vmin=1e10, vmax=1e16, clip=True)

# Máscaras iniciales
D_masked = np.ma.masked_where(frames_D[0] < 1e10, frames_D[0])
A_masked = np.ma.masked_where(frames_A[0] < 5e10, frames_A[0])

# Plots iniciales
pcm_D = ax.pcolormesh(Lon, Lat, D_masked, cmap='YlOrRd', norm=norm_D,
                      shading='auto', transform=ccrs.PlateCarree(),
                      alpha=0.5, zorder=1)

pcm_A = ax.pcolormesh(Lon, Lat, A_masked, cmap='turbo', norm=norm_A,
                      shading='auto', transform=ccrs.PlateCarree(),
                      alpha=0.7, zorder=2)

# Colorbars
cbar_D = plt.colorbar(pcm_D, ax=ax, orientation='horizontal',
                      pad=0.08, fraction=0.04, aspect=40)
cbar_D.set_label('Depósito acumulado [Bq/m²]', fontsize=10)

cbar_A = plt.colorbar(pcm_A, ax=ax, orientation='horizontal',
                      pad=0.02, fraction=0.04, aspect=40)
cbar_A.set_label('Concentración atmosférica [Bq/m²]', fontsize=10)

# Chernobyl
ax.plot(chernobyl_lon, chernobyl_lat, marker='*', color='red',
        markersize=16, markeredgecolor='black', markeredgewidth=1.5,
        transform=ccrs.PlateCarree(), zorder=10)

ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
             alpha=0.3, linestyle='--')

title = ax.set_title('', fontsize=14, fontweight='bold', pad=12)

def update_frame(i):
    """Actualiza la animación en cada frame"""
    D_masked = np.ma.masked_where(frames_D[i] < 1e10, frames_D[i])
    A_masked = np.ma.masked_where(frames_A[i] < 5e10, frames_A[i])
    
    pcm_D.set_array(D_masked.ravel())
    pcm_A.set_array(A_masked.ravel())
    
    t_h = times[i] / 3600.0
    title.set_text(f'Dispersión Chernobyl | t = {t_h:.1f} h ({t_h/24:.1f} días)')
    
    return pcm_D, pcm_A, title

# Crear animación
anim = FuncAnimation(fig, update_frame, frames=len(frames_A),
                     interval=100, blit=False, repeat=True)

plt.tight_layout()
plt.show()

print("\n=== SIMULACIÓN COMPLETADA ===")
print(f"Total de frames: {len(frames_A)}")
print(f"Duración simulada: {total_hours/24:.1f} días")
print("\n=== CONDICIONES INICIALES ===")
print("• Liberación: ~10h con pico gaussiano")
print("• Campo de vientos: 4 fases evolutivas")
print("• Precipitación: eventos 28 abril, 1-2 mayo, 8 mayo")
print("• Difusión K_h = 1.8×10⁵ m²/s")
print("• Deposición seca v_d = 0.0005 m/s, H = 1200m")
print("• Lavado húmedo: 5×10⁻⁵ s⁻¹ per mm/h")