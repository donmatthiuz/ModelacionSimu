import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
sys.path.append('../parte1')
sys.path.append('../parte2')
from simulador import ParticleSystem, SIRSimulation
from simulador2 import CellularAutomatonSIR, SIRAutomatonSimulation

def sir_ode_rk4(beta, gamma, S0, I0, R0, N, dt, steps):
    S = np.empty(steps+1)
    I = np.empty(steps+1)
    R = np.empty(steps+1)
    S[0], I[0], R[0] = S0, I0, R0
    def deriv(s,i,r):
        ds = -beta * s * i / N
        di = beta * s * i / N - gamma * i
        dr = gamma * i
        return ds, di, dr
    for t in range(steps):
        s,i,r = S[t], I[t], R[t]
        ds1,di1,dr1 = deriv(s,i,r)
        ds2,di2,dr2 = deriv(s+0.5*ds1*(1), i+0.5*di1*(1), r+0.5*dr1*(1))
        ds3,di3,dr3 = deriv(s+0.5*ds2*(1), i+0.5*di2*(1), r+0.5*dr2*(1))
        ds4,di4,dr4 = deriv(s+ds3*(1), i+di3*(1), r+dr3*(1))
        S[t+1] = s + (ds1 + 2*ds2 + 2*ds3 + ds4)/6.0
        I[t+1] = i + (di1 + 2*di2 + 2*di3 + di4)/6.0
        R[t+1] = r + (dr1 + 2*dr2 + 2*dr3 + dr4)/6.0
    return S, I, R

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def run_particle_experiments(params, Nexp, seed):
    L = params['L']
    N = params['N']
    I0 = params['I0']
    vmax = params['vmax']
    r = params['r']
    beta = params['beta']
    gamma = params['gamma']
    dt = params['dt']
    steps = params['steps']
    np.random.seed(seed)
    initial_pos = np.random.rand(N,2)*L
    initial_vel = (np.random.rand(N,2)-0.5)*2*vmax
    infected_idx = np.random.choice(N, I0, replace=False)
    all_S, all_I, all_R = [], [], []
    for exp in range(Nexp):
        sys_exp = ParticleSystem(L,N,I0,vmax,r,beta,gamma,dt)
        sys_exp.pos = initial_pos.copy()
        sys_exp.vel = initial_vel.copy()
        sys_exp.state = np.zeros(N, dtype=int)
        sys_exp.state[infected_idx] = 1
        sys_exp.S_hist = []
        sys_exp.I_hist = []
        sys_exp.R_hist = []
        sys_exp.record_counts()
        for _ in range(steps):
            sys_exp.step()
        all_S.append(sys_exp.S_hist)
        all_I.append(sys_exp.I_hist)
        all_R.append(sys_exp.R_hist)
    all_S = np.array(all_S, dtype=float)
    all_I = np.array(all_I, dtype=float)
    all_R = np.array(all_R, dtype=float)
    return all_S.mean(axis=0), all_I.mean(axis=0), all_R.mean(axis=0), initial_pos, initial_vel, infected_idx

def run_automaton_experiments(params, Nexp, seed):
    M = params['M']
    N = params['N']
    I0 = params['I0']
    r = params['r']
    beta = params['beta']
    gamma = params['gamma']
    T = params['T']
    start_point = params.get('start_point', None)
    np.random.seed(seed)
    if start_point is None:
        center_m, center_n = M//2, N//2
    else:
        center_m, center_n = start_point
    initial_infected = []
    infected_count = 0
    span = int(np.ceil(np.sqrt(I0)))
    for dm in range(-span, span+1):
        for dn in range(-span, span+1):
            if infected_count >= I0:
                break
            m, n = center_m+dm, center_n+dn
            if 0<=m<M and 0<=n<N:
                initial_infected.append((m,n))
                infected_count += 1
        if infected_count >= I0:
            break
    all_S, all_I, all_R = [], [], []
    for exp in range(Nexp):
        ca = CellularAutomatonSIR(M,N,I0,r,beta,gamma,T,start_point)
        ca.grid = np.zeros((M,N), dtype=int)
        for (m,n) in initial_infected:
            ca.grid[m,n] = 1
        ca.S_hist = []
        ca.I_hist = []
        ca.R_hist = []
        ca.record_counts()
        for _ in range(T):
            ca.step()
        all_S.append(ca.S_hist)
        all_I.append(ca.I_hist)
        all_R.append(ca.R_hist)
    all_S = np.array(all_S, dtype=float)
    all_I = np.array(all_I, dtype=float)
    all_R = np.array(all_R, dtype=float)
    return all_S.mean(axis=0), all_I.mean(axis=0), all_R.mean(axis=0), initial_infected

def plot_and_save_mean(t, S_mean, I_mean, R_mean, ode_tuple, out_png):
    plt.figure(figsize=(9,5))
    plt.plot(t, S_mean, 'g-', label='S_mean')
    plt.plot(t, I_mean, 'y-', label='I_mean')
    plt.plot(t, R_mean, 'r-', label='R_mean')
    if ode_tuple is not None:
        Sode, Iode, Rode = ode_tuple
        plt.plot(t, Sode, '--', label='S_ODE')
        plt.plot(t, Iode, '--', label='I_ODE')
        plt.plot(t, Rode, '--', label='R_ODE')
    plt.xlabel('Tiempo')
    plt.ylabel('Número')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

def make_and_move_particle_gif(params, initial_pos, initial_vel, infected_idx, seed, outpath):
    L = params['L']
    N = params['N']
    I0 = params['I0']
    vmax = params['vmax']
    r = params['r']
    beta = params['beta']
    gamma = params['gamma']
    dt = params['dt']
    steps = params['steps']
    sim = SIRSimulation(L=L, N=N, I0=I0, vmax=vmax, r=r, beta=beta, gamma=gamma, dt=dt, steps=steps)
    sim.sys.pos = initial_pos.copy()
    sim.sys.vel = initial_vel.copy()
    sim.sys.state = np.zeros(N, dtype=int)
    sim.sys.state[infected_idx] = 1
    sim.sys.S_hist = []
    sim.sys.I_hist = []
    sim.sys.R_hist = []
    sim.sys.record_counts()
    sim.run(save_animation=True)
    files = ['sir_particles.gif','sir_curves.png','sir_final_state.png']
    for f in files:
        if os.path.exists(f):
            dst = outpath.replace('.gif','') + '_' + f
            shutil.move(f, dst)

def make_and_move_automaton_gif(params, initial_infected, seed, outpath):
    M = params['M']
    N = params['N']
    I0 = params['I0']
    r = params['r']
    beta = params['beta']
    gamma = params['gamma']
    T = params['T']
    start_point = params.get('start_point', None)
    sim = SIRAutomatonSimulation(M=M, N=N, I0=I0, r=r, beta=beta, gamma=gamma, T=T, start_point=start_point)
    sim.ca.grid = np.zeros((M,N), dtype=int)
    for (m,n) in initial_infected:
        sim.ca.grid[m,n] = 1
    sim.ca.S_hist = []
    sim.ca.I_hist = []
    sim.ca.R_hist = []
    sim.ca.record_counts()
    sim.run(save_animation=True)
    files = ['sir_automaton.gif','sir_automaton_curves.png','sir_automaton_final.png']
    for f in files:
        if os.path.exists(f):
            dst = outpath.replace('.gif','') + '_' + f
            shutil.move(f, dst)

def main():
    base_seed = 33 
    Nexp = 30 
    particle_param_sets = [
        {'L':10.0,'N':200,'I0':5,'vmax':0.8,'r':0.3,'beta':1.5,'gamma':0.05,'dt':0.1,'steps':250},
        {'L':10.0,'N':200,'I0':5,'vmax':0.4,'r':0.3,'beta':0.8,'gamma':0.05,'dt':0.1,'steps':250},
        {'L':12.0,'N':300,'I0':10,'vmax':0.8,'r':0.25,'beta':1.2,'gamma':0.06,'dt':0.1,'steps':250}
    ]
    automaton_param_sets = [
        {'M':60,'N':60,'I0':10,'r':2,'beta':0.4,'gamma':0.05,'T':80,'start_point':(15,15)},
        {'M':60,'N':60,'I0':10,'r':1,'beta':0.3,'gamma':0.05,'T':80,'start_point':None},
        {'M':80,'N':80,'I0':20,'r':2,'beta':0.5,'gamma':0.06,'T':90,'start_point':(40,40)}
    ]
    for i,p in enumerate(particle_param_sets, start=1):
        seed = base_seed + i
        S_mean, I_mean, R_mean, pos0, vel0, infected_idx = run_particle_experiments(p, Nexp, seed)
        t = np.arange(len(S_mean)) * p['dt']
        S0 = p['N'] - p['I0']
        Sode, Iode, Rode = sir_ode_rk4(p['beta'], p['gamma'], S0, p['I0'], 0, p['N'], p['dt'], p['steps'])
        out_png = f"../particles_avg_set{i}.png"
        plot_and_save_mean(t, S_mean, I_mean, R_mean, (Sode,Iode,Rode), out_png)
        snap_out = f"../particles_set{i}.gif"
        make_and_move_particle_gif(p, pos0, vel0, infected_idx, seed, snap_out)
    for i,a in enumerate(automaton_param_sets, start=1):
        seed = base_seed + 100 + i
        S_mean, I_mean, R_mean, initial_infected = run_automaton_experiments(a, Nexp, seed)
        t = np.arange(len(S_mean))
        cells = a['M'] * a['N']
        Sode, Iode, Rode = sir_ode_rk4(a['beta'], a['gamma'], cells - a['I0'], a['I0'], 0, cells, 1.0, a['T'])
        out_png = f"../automaton_avg_set{i}.png"
        plot_and_save_mean(t, S_mean, I_mean, R_mean, (Sode,Iode,Rode), out_png)
        snap_out = f"../automaton_set{i}.gif"
        make_and_move_automaton_gif(a, initial_infected, seed, snap_out)
    print('Done')


main()
