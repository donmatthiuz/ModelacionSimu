import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from scipy.ndimage import gaussian_filter


class Domain:
    def __init__(self, lon_min=10.0, lon_max=80.0, lat_min=25.0, lat_max=70.0, nx=220, ny=160):
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.nx = nx
        self.ny = ny
        self.deg2m = 111000.0

        self.lons = np.linspace(lon_min, lon_max, nx)
        self.lats = np.linspace(lat_min, lat_max, ny)
        self.Lon, self.Lat = np.meshgrid(self.lons, self.lats)

        self.dx_m = np.mean(self.deg2m * np.cos(np.deg2rad(self.lats)))
        self.dy_m = self.deg2m


class ChernobylSource:
    def __init__(self, domain: Domain):
        self.scale = 5e13
        self.chernobyl_lon = 30.1
        self.chernobyl_lat = 51.3
        self.source_radius_deg = 0.25
        self.release_rate = 3.0 * self.scale
        self.release_duration = 10 * 3600.0
        self.source_shape = self._build_source_shape(domain.Lon, domain.Lat)

    def _build_source_shape(self, Lon, Lat):
        r = np.sqrt((Lon - self.chernobyl_lon) ** 2 + (Lat - self.chernobyl_lat) ** 2)
        mask = np.exp(-0.5 * (r / (self.source_radius_deg / 2.5)) ** 2)
        return mask / mask.sum()


class WindExtreme:
    def field(self, domain: Domain, t_seconds: float):
        Lon = domain.Lon
        Lat = domain.Lat
        t_h = t_seconds / 3600.0

        u_west = -3.5 * np.exp(-((Lat - 52.0) / 10.0) ** 2)
        u_east = 3.0 * np.exp(-((Lat - 48.0) / 9.0) ** 2)
        v_south = -2.8 * np.exp(-((Lon - 35.0) / 15.0) ** 2) * np.exp(-((Lat - 55.0) / 10.0) ** 2)
        v_north = 2.5 * np.exp(-((Lon - 25.0) / 12.0) ** 2) * np.exp(-((Lat - 48.0) / 8.0) ** 2)

        lon_low1 = 18.0 + t_h * 0.08
        lat_low1 = 60.0
        r1 = np.sqrt(((Lon - lon_low1) * 1.3) ** 2 + (Lat - lat_low1) ** 2)
        theta1 = np.arctan2(Lat - lat_low1, (Lon - lon_low1) * 1.3)
        V1 = 18.0 * np.exp(-(r1 / 15.0) ** 2)
        u_low1 = -V1 * np.sin(theta1)
        v_low1 = V1 * np.cos(theta1)

        lon_low2 = 22.0 + t_h * 0.05
        lat_low2 = 56.0
        r2 = np.sqrt(((Lon - lon_low2) * 1.4) ** 2 + (Lat - lat_low2) ** 2)
        theta2 = np.arctan2(Lat - lat_low2, (Lon - lon_low2) * 1.4)
        V2 = 15.0 * np.exp(-(r2 / 13.0) ** 2) * (0.5 + 0.5 * np.sin(t_h * 0.05))
        u_low2 = -V2 * np.sin(theta2)
        v_low2 = V2 * np.cos(theta2)

        lon_low3 = 26.0 + 5.0 * np.sin(t_h * 0.03)
        lat_low3 = 48.0 + 3.0 * np.cos(t_h * 0.04)
        r3 = np.sqrt(((Lon - lon_low3) * 1.2) ** 2 + (Lat - lat_low3) ** 2)
        theta3 = np.arctan2(Lat - lat_low3, (Lon - lon_low3) * 1.2)
        V3 = 16.0 * np.exp(-(r3 / 11.0) ** 2)
        u_low3 = -V3 * np.sin(theta3)
        v_low3 = V3 * np.cos(theta3)

        lon_low4 = 35.0 + t_h * 0.06
        lat_low4 = 45.0
        r4 = np.sqrt(((Lon - lon_low4) * 1.3) ** 2 + (Lat - lat_low4) ** 2)
        theta4 = np.arctan2(Lat - lat_low4, (Lon - lon_low4) * 1.3)
        V4 = 14.0 * np.exp(-(r4 / 12.0) ** 2)
        u_low4 = -V4 * np.sin(theta4)
        v_low4 = V4 * np.cos(theta4)

        shear_lat1 = 53.0 + 2.0 * np.sin(t_h * 0.04)
        shear_profile1 = np.tanh((Lat - shear_lat1) / 2.0)
        u_shear1 = 22.0 * shear_profile1
        v_shear1 = 5.0 * np.sin(2.0 * np.pi * (Lon - 20.0 + t_h * 0.3) / 25.0) * np.exp(
            -((Lat - shear_lat1) / 5.0) ** 2
        )

        shear_lat2 = 48.0 + 1.5 * np.cos(t_h * 0.05)
        shear_profile2 = np.tanh((Lat - shear_lat2) / 2.5)
        u_shear2 = 18.0 * shear_profile2
        v_shear2 = -4.5 * np.sin(2.0 * np.pi * (Lon - 30.0 - t_h * 0.2) / 30.0) * np.exp(
            -((Lat - shear_lat2) / 6.0) ** 2
        )

        u_shear3 = 12.0 * (Lat - 50.0) / 10.0
        v_shear3 = 9.0 * np.sin(2.0 * np.pi * Lon / 35.0)

        rot_center_lon = 28.0 + 3.0 * np.cos(t_h * 0.03)
        rot_center_lat = 51.0 + 2.0 * np.sin(t_h * 0.04)
        r_rot = np.sqrt(((Lon - rot_center_lon) * 1.2) ** 2 + (Lat - rot_center_lat) ** 2)
        theta_rot = np.arctan2(Lat - rot_center_lat, (Lon - rot_center_lon) * 1.2)
        omega_rot = 15.0 * np.exp(-(r_rot / 8.0) ** 2) * (1.0 + 0.5 * np.sin(t_h * 0.08))
        u_shear4 = -omega_rot * (Lat - rot_center_lat)
        v_shear4 = omega_rot * (Lon - rot_center_lon) * 1.2

        lon_def1 = 32.0 + 4.0 * np.cos(t_h * 0.02)
        lat_def1 = 50.0 + 3.0 * np.sin(t_h * 0.03)
        u_def1 = 12.0 * (Lon - lon_def1) * np.exp(
            -(((Lon - lon_def1) ** 2 + (Lat - lat_def1) ** 2) / 7.0**2)
        )
        v_def1 = 12.0 * (Lat - lat_def1) * np.exp(
            -(((Lon - lon_def1) ** 2 + (Lat - lat_def1) ** 2) / 7.0**2)
        )

        lon_def2 = 24.0 + 3.0 * np.sin(t_h * 0.04)
        lat_def2 = 54.0
        u_def2 = -9.5 * (Lon - lon_def2) * np.exp(
            -(((Lon - lon_def2) ** 2 + (Lat - lat_def2) ** 2) / 9.0**2)
        ) * np.cos(t_h * 0.1)
        v_def2 = 9.5 * (Lat - lat_def2) * np.exp(
            -(((Lon - lon_def2) ** 2 + (Lat - lat_def2) ** 2) / 9.0**2)
        ) * np.cos(t_h * 0.1)

        band_phase = 2.0 * np.pi * (Lon / 20.0 + t_h * 0.04)
        u_def3 = 10.0 * np.sin(band_phase) * np.exp(-((Lat - 51.0) / 7.0) ** 2)
        v_def3 = -7.0 * np.cos(band_phase) * np.exp(-((Lat - 51.0) / 7.0) ** 2)

        lon_hyp = 29.0
        lat_hyp = 52.0
        u_def4 = 8.0 * (Lon - lon_hyp) * np.exp(
            -(((Lon - lon_hyp) ** 2 + (Lat - lat_hyp) ** 2) / 12.0**2)
        )
        v_def4 = -8.0 * (Lat - lat_hyp) * np.exp(
            -(((Lon - lon_hyp) ** 2 + (Lat - lat_hyp) ** 2) / 12.0**2)
        )

        vort_phase = t_h * 0.05
        u_def5 = np.zeros_like(Lon)
        v_def5 = np.zeros_like(Lat)
        for i, (vlon, vlat) in enumerate([(20, 57), (27, 49), (35, 53)]):
            r_vort = np.sqrt(((Lon - vlon) ** 2 + (Lat - vlat) ** 2))
            theta_vort = np.arctan2(Lat - vlat, Lon - vlon)
            sign = (-1) ** i
            V_vort = sign * 7.0 * np.exp(-(r_vort / 6.0) ** 2) * np.sin(vort_phase + i * np.pi / 3)
            u_def5 += -V_vort * np.sin(theta_vort)
            v_def5 += V_vort * np.cos(theta_vort)

        turb_x1 = 2.0 * np.pi * (Lon / 8.0 + t_h * 0.02)
        turb_y1 = 2.0 * np.pi * (Lat / 7.0 - t_h * 0.015)
        u_turb1 = 5.5 * np.sin(turb_x1) * np.cos(turb_y1)
        v_turb1 = 5.5 * np.cos(turb_x1) * np.sin(turb_y1)

        turb_x2 = 2.0 * np.pi * (Lon / 12.0 - t_h * 0.025)
        turb_y2 = 2.0 * np.pi * (Lat / 9.0 + t_h * 0.018)
        u_turb2 = 4.0 * np.cos(turb_x2) * np.sin(turb_y2)
        v_turb2 = 4.0 * np.sin(turb_x2) * np.cos(turb_y2)

        turb_x3 = 2.0 * np.pi * (Lon / 15.0 + t_h * 0.03)
        u_turb3 = 3.5 * np.sin(turb_x3 + np.sin(turb_y1))
        v_turb3 = 3.5 * np.cos(turb_y1 + np.cos(turb_x3))

        wave_phase1 = 2.0 * np.pi * (Lon - 15.0 + t_h * 0.25) / 40.0
        u_wave1 = 7.0 * np.sin(wave_phase1) * np.exp(-((Lat - 56.0) / 10.0) ** 2)
        v_wave1 = -3.5 * np.cos(wave_phase1) * np.exp(-((Lat - 56.0) / 10.0) ** 2)

        wave_phase2 = 2.0 * np.pi * (Lon - 25.0 - t_h * 0.2) / 35.0
        u_wave2 = 5.5 * np.cos(wave_phase2) * np.exp(-((Lat - 48.0) / 8.0) ** 2)
        v_wave2 = 4.0 * np.sin(wave_phase2) * np.exp(-((Lat - 48.0) / 8.0) ** 2)

        jet_lat = 54.0 + 4.0 * np.sin(2.0 * np.pi * t_h / 50.0)
        u_jet = 10.0 * np.exp(-((Lat - jet_lat) / 6.0) ** 2)
        v_jet = 2.5 * np.sin(2.0 * np.pi * (Lon - 25.0 + t_h * 0.3) / 30.0) * np.exp(
            -((Lat - jet_lat) / 8.0) ** 2
        )

        u = (
            u_west
            + u_east
            + u_low1
            + u_low2
            + u_low3
            + u_low4
            + u_shear1
            + u_shear2
            + u_shear3
            + u_shear4
            + u_def1
            + u_def2
            + u_def3
            + u_def4
            + u_def5
            + u_turb1
            + u_turb2
            + u_turb3
            + u_wave1
            + u_wave2
            + u_jet
        )

        v = (
            v_south
            + v_north
            + v_low1
            + v_low2
            + v_low3
            + v_low4
            + v_shear1
            + v_shear2
            + v_shear3
            + v_shear4
            + v_def1
            + v_def2
            + v_def3
            + v_def4
            + v_def5
            + v_turb1
            + v_turb2
            + v_turb3
            + v_wave1
            + v_wave2
            + v_jet
        )

        return u, v


class PrecipitationModel:
    def field(self, domain: Domain, t: float):
        Lon = domain.Lon
        Lat = domain.Lat
        t_h = t / 3600.0
        rain = np.zeros_like(Lon)

        if 42 <= t_h <= 54:
            r1 = np.sqrt((Lon - 16.0) ** 2 + (Lat - 61.0) ** 2)
            rain += 6.0 * np.exp(-(r1 / 9.0) ** 2)

        if 90 <= t_h <= 126:
            r2 = np.sqrt((Lon - 14.0) ** 2 + (Lat - 50.0) ** 2)
            rain += 4.5 * np.exp(-(r2 / 8.0) ** 2)
            r3 = np.sqrt((Lon - 20.0) ** 2 + (Lat - 52.0) ** 2)
            rain += 3.5 * np.exp(-(r3 / 7.0) ** 2)

        if 282 <= t_h <= 294:
            r4 = np.sqrt((Lon - 13.0) ** 2 + (Lat - 58.0) ** 2)
            rain += 8.0 * np.exp(-(r4 / 6.0) ** 2)

        if t_h > 150:
            lon_front = 18.0 + ((t_h - 150.0) / 24.0) * 10.0
            r5 = np.sqrt(((Lon - lon_front) / 2.5) ** 2 + (Lat - 54.0) ** 2)
            rain += 3.0 * np.exp(-(r5 / 7.0) ** 2)

        return rain


class PhysicsOperators:
    def __init__(self, domain: Domain, source: ChernobylSource):
        self.domain = domain
        self.source = source
        self.decay_lambda = np.log(2) / (30.17 * 365.25 * 24 * 3600)
        v_d = 0.0005
        mixed_layer_h = 1200.0
        self.alpha = v_d / mixed_layer_h
        self.scavenging_coeff = 5e-5

    def bilinear_interp_on_grid(self, A, lon_pts, lat_pts):
        Lon = self.domain.Lon
        Lat = self.domain.Lat
        ny, nx = A.shape
        lon_vec = Lon[0, :]
        lat_vec = Lat[:, 0]
        dx_deg = lon_vec[1] - lon_vec[0]
        dy_deg = lat_vec[1] - lat_vec[0]

        ix = np.clip((lon_pts - lon_vec[0]) / dx_deg, 0, nx - 1.000001)
        iy = np.clip((lat_pts - lat_vec[0]) / dy_deg, 0, ny - 1.000001)

        i0 = np.floor(ix).astype(int)
        j0 = np.floor(iy).astype(int)
        i1 = np.clip(i0 + 1, 0, nx - 1)
        j1 = np.clip(j0 + 1, 0, ny - 1)

        wx = ix - i0
        wy = iy - j0

        return (
            (1 - wx) * (1 - wy) * A[j0, i0]
            + wx * (1 - wy) * A[j0, i1]
            + (1 - wx) * wy * A[j1, i0]
            + wx * wy * A[j1, i1]
        )

    def semi_lagrangian_step(self, A, u, v, dt):
        Lon = self.domain.Lon
        Lat = self.domain.Lat
        coslat = np.cos(np.deg2rad(Lat))
        lon_dep = Lon - (u * dt) / (self.domain.deg2m * coslat)
        lat_dep = Lat - (v * dt) / self.domain.deg2m
        return self.bilinear_interp_on_grid(A, lon_dep, lat_dep)

    def horizontal_diffusion(self, A, K_h, dt):
        sigma_m = np.sqrt(2.0 * K_h * dt)
        sigma_pix = sigma_m / np.mean([self.domain.dx_m, self.domain.dy_m])
        if sigma_pix < 0.1:
            return A
        A_diff = gaussian_filter(A, sigma=sigma_pix, mode="constant", cval=0)
        return A_diff * (A.sum() / (A_diff.sum() + 1e-30))

    def apply_source_sink(self, A, dt, t, precip):
        W = self.scavenging_coeff * precip
        kappa = self.alpha + self.decay_lambda + W
        expf = np.exp(-kappa * dt)

        depo_rate = self.alpha + W
        deposited = np.where(
            kappa > 1e-12,
            A * (depo_rate / kappa) * (1.0 - expf),
            A * depo_rate * dt,
        )

        source = np.zeros_like(A)
        if t < self.source.release_duration:
            t_h = t / 3600.0
            temporal = np.exp(-((t_h - 1.0) ** 2) / 8.0)
            source = self.source.release_rate * temporal * self.source.source_shape

        A_new = A * expf
        A_new = np.where(
            kappa > 1e-12,
            A_new + (source / kappa) * (1.0 - expf),
            A_new + source * dt,
        )

        return A_new, deposited

    def advance_timestep(self, A, D, t, dt, wind: WindExtreme, precip_model: PrecipitationModel):
        u, v = wind.field(self.domain, t)
        A_adv = self.semi_lagrangian_step(A, u, v, dt)
        A_diff = self.horizontal_diffusion(A_adv, K_h=2.5e5, dt=dt)
        precip = precip_model.field(self.domain, t)
        A_next, deposited = self.apply_source_sink(A_diff, dt, t, precip)
        D_next = D + deposited
        return A_next, D_next


class SimulationEngine:
    def __init__(
        self,
        domain: Domain,
        source: ChernobylSource,
        wind: WindExtreme,
        precip: PrecipitationModel,
        physics: PhysicsOperators,
    ):
        self.domain = domain
        self.source = source
        self.wind = wind
        self.precip = precip
        self.physics = physics

    def run(self, total_hours=240, dt=1800.0, max_frames=100):
        ny, nx = self.domain.ny, self.domain.nx
        A = np.zeros((ny, nx)) + 1e9
        D = np.zeros((ny, nx))

        nsteps = int((total_hours * 3600) / dt)
        frame_interval = max(1, nsteps // max_frames)

        frames_A = []
        frames_D = []
        times = []

        t = 0.0
        t0 = time.time()
        print(f"Simulación: {total_hours/24:.0f} días, {nsteps} pasos")

        for step in range(nsteps):
            A, D = self.physics.advance_timestep(A, D, t, dt, self.wind, self.precip)
            t += dt

            if step % frame_interval == 0 or step == nsteps - 1:
                frames_A.append(A.copy())
                frames_D.append(D.copy())
                times.append(t)

            if step % 20 == 0:
                print(f"  {step}/{nsteps} | t={t/3600:.1f}h | {time.time()-t0:.0f}s")

        print("Simulación completada.")
        return frames_A, frames_D, times


class Animator:
    def __init__(self, domain: Domain, source: ChernobylSource, frames_A, frames_D, times):
        self.domain = domain
        self.source = source
        self.frames_A = frames_A
        self.frames_D = frames_D
        self.times = times

        self.fig = plt.figure(figsize=(16, 8))
        self.ax = plt.axes(projection=ccrs.PlateCarree())
        self.ax.set_extent(
            [domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max],
            crs=ccrs.PlateCarree(),
        )

        self.ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.88", zorder=0)
        self.ax.add_feature(
            cfeature.OCEAN.with_scale("50m"),
            facecolor="lightblue",
            alpha=0.4,
            zorder=0,
        )
        self.ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, zorder=3)
        self.ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=3)

        self.norm_D = colors.LogNorm(vmin=1e9, vmax=5e14, clip=True)
        self.norm_A = colors.LogNorm(vmin=1e10, vmax=1e16, clip=True)

        D0_masked = np.ma.masked_where(self.frames_D[0] < 1e10, self.frames_D[0])
        A0_masked = np.ma.masked_where(self.frames_A[0] < 5e10, self.frames_A[0])

        self.pcm_D = self.ax.pcolormesh(
            self.domain.Lon,
            self.domain.Lat,
            D0_masked,
            cmap="YlOrRd",
            norm=self.norm_D,
            shading="auto",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            zorder=1,
        )

        self.pcm_A = self.ax.pcolormesh(
            self.domain.Lon,
            self.domain.Lat,
            A0_masked,
            cmap="turbo",
            norm=self.norm_A,
            shading="auto",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
            zorder=2,
        )

        cbar_D = plt.colorbar(
            self.pcm_D,
            ax=self.ax,
            orientation="horizontal",
            pad=0.08,
            fraction=0.04,
            aspect=40,
        )
        cbar_D.set_label("Depósito acumulado [Bq/m²]", fontsize=10)

        cbar_A = plt.colorbar(
            self.pcm_A,
            ax=self.ax,
            orientation="horizontal",
            pad=0.02,
            fraction=0.04,
            aspect=40,
        )
        cbar_A.set_label("Concentración atmosférica [Bq/m²]", fontsize=10)

        self.ax.plot(
            self.source.chernobyl_lon,
            self.source.chernobyl_lat,
            marker="*",
            color="red",
            markersize=16,
            markeredgecolor="black",
            markeredgewidth=1.5,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

        self.ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.3,
            linestyle="--",
        )

        self.title = self.ax.set_title("", fontsize=14, fontweight="bold", pad=12)

    def _update(self, i):
        D_masked = np.ma.masked_where(self.frames_D[i] < 1e10, self.frames_D[i])
        A_masked = np.ma.masked_where(self.frames_A[i] < 5e10, self.frames_A[i])

        self.pcm_D.set_array(D_masked.ravel())
        self.pcm_A.set_array(A_masked.ravel())

        t_h = self.times[i] / 3600.0
        self.title.set_text(
            f"Dispersión Chernobyl - FILAMENTACIÓN EXTREMA | t = {t_h:.1f} h ({t_h/24:.1f} días)"
        )

        return self.pcm_D, self.pcm_A, self.title

    def animate(self, interval=100, repeat=True):
        anim = FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.frames_A),
            interval=interval,
            blit=False,
            repeat=repeat,
        )
        plt.tight_layout()
        plt.show()
        return anim


if __name__ == "__main__":
    domain = Domain()
    source = ChernobylSource(domain)
    wind = WindExtreme()
    precip = PrecipitationModel()
    physics = PhysicsOperators(domain, source)

    engine = SimulationEngine(domain, source, wind, precip, physics)
    frames_A, frames_D, times = engine.run(total_hours=240, dt=1800.0, max_frames=100)

    animator = Animator(domain, source, frames_A, frames_D, times)
    animator.animate()

    print("\n=== CONFIGURACIÓN DE FILAMENTACIÓN EXTREMA ===")
    print("✓ Flujos de fondo débiles y balanceados")
    print("✓ 4 sistemas ciclónicos intensos y competitivos")
    print("✓ 4 frentes de cizalladura superpuestos")
    print("✓ 5 zonas de deformación activas")
    print("✓ 3 campos de turbulencia amplificada")
    print("✓ Difusión aumentada: K_h = 2.5×10⁵ m²/s")
    print("✓ Resultado: dispersión multidireccional con filamentos intensos")
