"""
Simulación Chernobyl con FILAMENTACIÓN EXTREMA
- Vientos fuertes al sur y oeste desde el inicio
- Turbulencia amplificada
- Múltiples campos de deformación para fragmentación rápida
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
# PARÁMETROS FÍSICOS
# -----------------------
SCALE = 5e13
decay_lambda = np.log(2) / (30.17 * 365.25 * 24 * 3600)
v_d = 0.0005
mixed_layer_h = 1200.0
alpha = v_d / mixed_layer_h
scavenging_coeff = 5e-5

# -----------------------
# FUENTE
# -----------------------
chernobyl_lon = 30.1
chernobyl_lat = 51.3
source_radius_deg = 0.25
release_rate = 3.0 * SCALE
release_duration = 10 * 3600.0

def source_spatial_mask(Lon, Lat, lon0, lat0, radius_deg):
    r = np.sqrt((Lon - lon0)**2 + (Lat - lat0)**2)
    mask = np.exp(-0.5 * (r / (radius_deg/2.5))**2)
    return mask / mask.sum()

source_shape = source_spatial_mask(Lon, Lat, chernobyl_lon, chernobyl_lat, source_radius_deg)

# -----------------------
# CAMPO DE VIENTOS CON FILAMENTACIÓN EXTREMA
# -----------------------
def wind_field_extreme_filaments(Lon, Lat, t_seconds):
    """
    Campo de vientos optimizado para filamentación rápida:
    - Vientos fuertes al OESTE y SUR desde el inicio
    - Múltiples zonas de cizalladura
    - Turbulencia amplificada
    - Campos de deformación intensos
    """
    t_h = t_seconds / 3600.0
    
    # === FLUJOS DE FONDO DÉBILES Y MULTIDIRECCIONALES ===
    # En lugar de un flujo fuerte unidireccional, creamos varios débiles que compiten
    
    # Flujo débil al oeste (no dominante)
    u_west = -3.5 * np.exp(-((Lat - 52.0)/10.0)**2)
    
    # Flujo débil al este (contrarresta al oeste)
    u_east = 3.0 * np.exp(-((Lat - 48.0)/9.0)**2)
    
    # Flujo débil al sur
    v_south = -2.8 * np.exp(-((Lon - 35.0)/15.0)**2) * np.exp(-((Lat - 55.0)/10.0)**2)
    
    # Flujo débil al norte (contrarresta al sur)
    v_north = 2.5 * np.exp(-((Lon - 25.0)/12.0)**2) * np.exp(-((Lat - 48.0)/8.0)**2)
    
    # === SISTEMAS CICLÓNICOS BALANCEADOS (dispersión multidireccional) ===
    
    # Baja 1: Escandinavia (flujo N-NE) - MÁS INTENSA
    lon_low1 = 18.0 + t_h * 0.08
    lat_low1 = 60.0
    r1 = np.sqrt(((Lon - lon_low1)*1.3)**2 + (Lat - lat_low1)**2)
    theta1 = np.arctan2(Lat - lat_low1, (Lon - lon_low1)*1.3)
    V1 = 18.0 * np.exp(-(r1/15.0)**2)  # Aumentado
    u_low1 = -V1 * np.sin(theta1)
    v_low1 = V1 * np.cos(theta1)
    
    # Baja 2: Mar Báltico (flujo E-SE) - MÁS INTENSA
    lon_low2 = 22.0 + t_h * 0.05
    lat_low2 = 56.0
    r2 = np.sqrt(((Lon - lon_low2)*1.4)**2 + (Lat - lat_low2)**2)
    theta2 = np.arctan2(Lat - lat_low2, (Lon - lon_low2)*1.4)
    V2 = 15.0 * np.exp(-(r2/13.0)**2) * (0.5 + 0.5 * np.sin(t_h * 0.05))  # Aumentado
    u_low2 = -V2 * np.sin(theta2)
    v_low2 = V2 * np.cos(theta2)
    
    # Baja 3: Europa Central (movimiento errático) - MÁS INTENSA
    lon_low3 = 26.0 + 5.0 * np.sin(t_h * 0.03)
    lat_low3 = 48.0 + 3.0 * np.cos(t_h * 0.04)
    r3 = np.sqrt(((Lon - lon_low3)*1.2)**2 + (Lat - lat_low3)**2)
    theta3 = np.arctan2(Lat - lat_low3, (Lon - lon_low3)*1.2)
    V3 = 16.0 * np.exp(-(r3/11.0)**2)  # Aumentado
    u_low3 = -V3 * np.sin(theta3)
    v_low3 = V3 * np.cos(theta3)
    
    # Baja 4: Mar Negro (flujo S-SW) - NUEVA para dispersión sur
    lon_low4 = 35.0 + t_h * 0.06
    lat_low4 = 45.0
    r4 = np.sqrt(((Lon - lon_low4)*1.3)**2 + (Lat - lat_low4)**2)
    theta4 = np.arctan2(Lat - lat_low4, (Lon - lon_low4)*1.3)
    V4 = 14.0 * np.exp(-(r4/12.0)**2)
    u_low4 = -V4 * np.sin(theta4)
    v_low4 = V4 * np.cos(theta4)
    
    # === CIZALLADURA EXTREMA (GENERA FILAMENTOS) ===
    
    # Frente 1: Zona de cizalladura W-E (lat ~53°N) - MÁS INTENSA
    shear_lat1 = 53.0 + 2.0 * np.sin(t_h * 0.04)
    shear_profile1 = np.tanh((Lat - shear_lat1) / 2.0)  # Más abrupto
    u_shear1 = 22.0 * shear_profile1  # Aumentado
    v_shear1 = 5.0 * np.sin(2.0 * np.pi * (Lon - 20.0 + t_h * 0.3) / 25.0) * np.exp(-((Lat - shear_lat1)/5.0)**2)
    
    # Frente 2: Cizalladura secundaria (lat ~48°N) - MÁS INTENSA
    shear_lat2 = 48.0 + 1.5 * np.cos(t_h * 0.05)
    shear_profile2 = np.tanh((Lat - shear_lat2) / 2.5)
    u_shear2 = 18.0 * shear_profile2  # Aumentado
    v_shear2 = -4.5 * np.sin(2.0 * np.pi * (Lon - 30.0 - t_h * 0.2) / 30.0) * np.exp(-((Lat - shear_lat2)/6.0)**2)
    
    # Frente 3: Cizalladura vertical (gradiente fuerte N-S) - MÁS INTENSA
    u_shear3 = 12.0 * (Lat - 50.0) / 10.0  # Aumentado
    v_shear3 = 9.0 * np.sin(2.0 * np.pi * Lon / 35.0)  # Aumentado
    
    # Frente 4: Cizalladura rotatoria (NUEVA) - CREA FILAMENTOS ESPIRALES
    rot_center_lon = 28.0 + 3.0 * np.cos(t_h * 0.03)
    rot_center_lat = 51.0 + 2.0 * np.sin(t_h * 0.04)
    r_rot = np.sqrt(((Lon - rot_center_lon)*1.2)**2 + (Lat - rot_center_lat)**2)
    theta_rot = np.arctan2(Lat - rot_center_lat, (Lon - rot_center_lon)*1.2)
    
    # Rotación diferencial (más rápido cerca del centro)
    omega_rot = 15.0 * np.exp(-(r_rot/8.0)**2) * (1.0 + 0.5 * np.sin(t_h * 0.08))
    u_shear4 = -omega_rot * (Lat - rot_center_lat)
    v_shear4 = omega_rot * (Lon - rot_center_lon) * 1.2
    
    # === CAMPOS DE DEFORMACIÓN INTENSIFICADOS (ESTIRAMIENTO Y COMPRESIÓN) ===
    
    # Zona de estiramiento 1: Divergencia radial - MÁS INTENSA
    lon_def1 = 32.0 + 4.0 * np.cos(t_h * 0.02)
    lat_def1 = 50.0 + 3.0 * np.sin(t_h * 0.03)
    u_def1 = 12.0 * (Lon - lon_def1) * np.exp(-(((Lon-lon_def1)**2 + (Lat-lat_def1)**2) / 7.0**2))
    v_def1 = 12.0 * (Lat - lat_def1) * np.exp(-(((Lon-lon_def1)**2 + (Lat-lat_def1)**2) / 7.0**2))
    
    # Zona de estiramiento 2: Compresión-expansión alterna - MÁS INTENSA
    lon_def2 = 24.0 + 3.0 * np.sin(t_h * 0.04)
    lat_def2 = 54.0
    u_def2 = -9.5 * (Lon - lon_def2) * np.exp(-(((Lon-lon_def2)**2 + (Lat-lat_def2)**2) / 9.0**2)) * np.cos(t_h * 0.1)
    v_def2 = 9.5 * (Lat - lat_def2) * np.exp(-(((Lon-lon_def2)**2 + (Lat-lat_def2)**2) / 9.0**2)) * np.cos(t_h * 0.1)
    
    # Zona de estiramiento 3: Bandas de deformación - MÁS INTENSA
    band_phase = 2.0 * np.pi * (Lon / 20.0 + t_h * 0.04)
    u_def3 = 10.0 * np.sin(band_phase) * np.exp(-((Lat - 51.0)/7.0)**2)
    v_def3 = -7.0 * np.cos(band_phase) * np.exp(-((Lat - 51.0)/7.0)**2)
    
    # Zona de estiramiento 4: Campo hiperbólico (NUEVA) - GENERA FILAMENTOS LARGOS
    lon_hyp = 29.0
    lat_hyp = 52.0
    u_def4 = 8.0 * (Lon - lon_hyp) * np.exp(-(((Lon-lon_hyp)**2 + (Lat-lat_hyp)**2) / 12.0**2))
    v_def4 = -8.0 * (Lat - lat_hyp) * np.exp(-(((Lon-lon_hyp)**2 + (Lat-lat_hyp)**2) / 12.0**2))
    
    # Zona de estiramiento 5: Vórtices múltiples (NUEVA) - FRAGMENTACIÓN
    vort_phase = t_h * 0.05
    for i, (vlon, vlat) in enumerate([(20, 57), (27, 49), (35, 53)]):
        r_vort = np.sqrt(((Lon - vlon)**2 + (Lat - vlat)**2))
        theta_vort = np.arctan2(Lat - vlat, Lon - vlon)
        sign = (-1) ** i  # Alternar rotación
        V_vort = sign * 7.0 * np.exp(-(r_vort/6.0)**2) * np.sin(vort_phase + i * np.pi/3)
        if i == 0:
            u_def5 = -V_vort * np.sin(theta_vort)
            v_def5 = V_vort * np.cos(theta_vort)
        else:
            u_def5 += -V_vort * np.sin(theta_vort)
            v_def5 += V_vort * np.cos(theta_vort)
    
    # === TURBULENCIA DE MESOESCALA AMPLIFICADA ===
    
    # Turbulencia 1: Alta frecuencia espacial
    turb_x1 = 2.0 * np.pi * (Lon / 8.0 + t_h * 0.02)
    turb_y1 = 2.0 * np.pi * (Lat / 7.0 - t_h * 0.015)
    u_turb1 = 5.5 * np.sin(turb_x1) * np.cos(turb_y1)
    v_turb1 = 5.5 * np.cos(turb_x1) * np.sin(turb_y1)
    
    # Turbulencia 2: Vórtices pequeños
    turb_x2 = 2.0 * np.pi * (Lon / 12.0 - t_h * 0.025)
    turb_y2 = 2.0 * np.pi * (Lat / 9.0 + t_h * 0.018)
    u_turb2 = 4.0 * np.cos(turb_x2) * np.sin(turb_y2)
    v_turb2 = 4.0 * np.sin(turb_x2) * np.cos(turb_y2)
    
    # Turbulencia 3: Ondulaciones irregulares
    turb_x3 = 2.0 * np.pi * (Lon / 15.0 + t_h * 0.03)
    u_turb3 = 3.5 * np.sin(turb_x3 + np.sin(turb_y1))
    v_turb3 = 3.5 * np.cos(turb_y1 + np.cos(turb_x3))
    
    # === ONDAS PLANETARIAS (dispersión gran escala) ===
    wave_phase1 = 2.0 * np.pi * (Lon - 15.0 + t_h * 0.25) / 40.0
    u_wave1 = 7.0 * np.sin(wave_phase1) * np.exp(-((Lat - 56.0)/10.0)**2)
    v_wave1 = -3.5 * np.cos(wave_phase1) * np.exp(-((Lat - 56.0)/10.0)**2)
    
    wave_phase2 = 2.0 * np.pi * (Lon - 25.0 - t_h * 0.2) / 35.0
    u_wave2 = 5.5 * np.cos(wave_phase2) * np.exp(-((Lat - 48.0)/8.0)**2)
    v_wave2 = 4.0 * np.sin(wave_phase2) * np.exp(-((Lat - 48.0)/8.0)**2)
    
    # === CORRIENTE EN CHORRO (ondulante) ===
    jet_lat = 54.0 + 4.0 * np.sin(2.0 * np.pi * t_h / 50.0)
    u_jet = 10.0 * np.exp(-((Lat - jet_lat)/6.0)**2)
    v_jet = 2.5 * np.sin(2.0 * np.pi * (Lon - 25.0 + t_h * 0.3) / 30.0) * np.exp(-((Lat - jet_lat)/8.0)**2)
    
    # === SUPERPOSICIÓN TOTAL BALANCEADA ===
    u = (u_west + u_east + u_low1 + u_low2 + u_low3 + u_low4 +
         u_shear1 + u_shear2 + u_shear3 + u_shear4 +
         u_def1 + u_def2 + u_def3 + u_def4 + u_def5 +
         u_turb1 + u_turb2 + u_turb3 +
         u_wave1 + u_wave2 + u_jet)
    
    v = (v_south + v_north + v_low1 + v_low2 + v_low3 + v_low4 +
         v_shear1 + v_shear2 + v_shear3 + v_shear4 +
         v_def1 + v_def2 + v_def3 + v_def4 + v_def5 +
         v_turb1 + v_turb2 + v_turb3 +
         v_wave1 + v_wave2 + v_jet)
    
    return u, v

# -----------------------
# PRECIPITACIÓN
# -----------------------
def precipitation_field(Lon, Lat, t):
    t_h = t / 3600.0
    rain = np.zeros_like(Lon)
    
    # Eventos de precipitación
    if 42 <= t_h <= 54:
        r1 = np.sqrt((Lon - 16.0)**2 + (Lat - 61.0)**2)
        rain += 6.0 * np.exp(-(r1/9.0)**2)
    
    if 90 <= t_h <= 126:
        r2 = np.sqrt((Lon - 14.0)**2 + (Lat - 50.0)**2)
        rain += 4.5 * np.exp(-(r2/8.0)**2)
        r3 = np.sqrt((Lon - 20.0)**2 + (Lat - 52.0)**2)
        rain += 3.5 * np.exp(-(r3/7.0)**2)
    
    if 282 <= t_h <= 294:
        r4 = np.sqrt((Lon - 13.0)**2 + (Lat - 58.0)**2)
        rain += 8.0 * np.exp(-(r4/6.0)**2)
    
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
    u, v = wind_field_extreme_filaments(Lon, Lat, t)
    A_adv = semi_lagrangian_step(A, u, v, dt)
    A_diff = horizontal_diffusion(A_adv, K_h=2.5e5, dt=dt)  # Difusión aumentada
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
dt = 1800.0
total_hours = 240
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
# ANIMACIÓN
# -----------------------
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='0.88', zorder=0)
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue', alpha=0.4, zorder=0)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=3)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4, zorder=3)

norm_D = colors.LogNorm(vmin=1e9, vmax=5e14, clip=True)
norm_A = colors.LogNorm(vmin=1e10, vmax=1e16, clip=True)

D_masked = np.ma.masked_where(frames_D[0] < 1e10, frames_D[0])
A_masked = np.ma.masked_where(frames_A[0] < 5e10, frames_A[0])

pcm_D = ax.pcolormesh(Lon, Lat, D_masked, cmap='YlOrRd', norm=norm_D,
                      shading='auto', transform=ccrs.PlateCarree(),
                      alpha=0.5, zorder=1)

pcm_A = ax.pcolormesh(Lon, Lat, A_masked, cmap='turbo', norm=norm_A,
                      shading='auto', transform=ccrs.PlateCarree(),
                      alpha=0.7, zorder=2)

cbar_D = plt.colorbar(pcm_D, ax=ax, orientation='horizontal',
                      pad=0.08, fraction=0.04, aspect=40)
cbar_D.set_label('Depósito acumulado [Bq/m²]', fontsize=10)

cbar_A = plt.colorbar(pcm_A, ax=ax, orientation='horizontal',
                      pad=0.02, fraction=0.04, aspect=40)
cbar_A.set_label('Concentración atmosférica [Bq/m²]', fontsize=10)

ax.plot(chernobyl_lon, chernobyl_lat, marker='*', color='red',
        markersize=16, markeredgecolor='black', markeredgewidth=1.5,
        transform=ccrs.PlateCarree(), zorder=10)

ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
             alpha=0.3, linestyle='--')

title = ax.set_title('', fontsize=14, fontweight='bold', pad=12)

def update_frame(i):
    D_masked = np.ma.masked_where(frames_D[i] < 1e10, frames_D[i])
    A_masked = np.ma.masked_where(frames_A[i] < 5e10, frames_A[i])
    
    pcm_D.set_array(D_masked.ravel())
    pcm_A.set_array(A_masked.ravel())
    
    t_h = times[i] / 3600.0
    title.set_text(f'Dispersión Chernobyl - FILAMENTACIÓN EXTREMA | t = {t_h:.1f} h ({t_h/24:.1f} días)')
    
    return pcm_D, pcm_A, title

anim = FuncAnimation(fig, update_frame, frames=len(frames_A),
                     interval=100, blit=False, repeat=True)

plt.tight_layout()
plt.show()

print("\n=== CONFIGURACIÓN DE FILAMENTACIÓN EXTREMA ===")
print("✓ Flujos de fondo débiles y balanceados (sin dirección dominante)")
print("✓ 4 sistemas ciclónicos intensos y competitivos")
print("✓ 4 frentes de cizalladura superpuestos (incluye rotación)")
print("✓ 5 zonas de deformación activas (divergencia, compresión, vórtices)")
print("✓ 3 campos de turbulencia amplificada")
print("✓ Difusión aumentada: K_h = 2.5×10⁵ m²/s")
print("✓ RESULTADO: Dispersión multidireccional con filamentos intensos")