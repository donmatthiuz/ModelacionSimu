import argparse
from pathlib import Path
from tsplib import load_tsplib_2d
from tsp_utils import make_invented_instance, save_tsplib
from lp_tsp import solve_tsp_mtz

def distance_matrix(coords):
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            if i != j:
                xj, yj = coords[j]
                D[i][j] = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
    return D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eil", default="data/eil101.tsp")
    ap.add_argument("--inventado-n", type=int, default=80)
    args = ap.parse_args()
    
    # LP inventado con el tsp utils 
    inv_coords = make_invented_instance(args.inventado_n, seed=33)
    save_tsplib(inv_coords, "data/custom_inventado.tsp", name="INVENTADO")
    Dinv = distance_matrix(inv_coords)
    lp_res2 = solve_tsp_mtz(Dinv, time_limit=300)
    print(f"[inventado] Status={lp_res2.status}, Distancia={lp_res2.length:.2f}, Tiempo={lp_res2.elapsed:.2f}s")

    # LP en eil101 
    eil = load_tsplib_2d(args.eil)
    D = distance_matrix(eil.coords)
    lp_res = solve_tsp_mtz(D)
    print(f"[eil101] Status={lp_res.status}, Distancia={lp_res.length:.2f}, Tiempo={lp_res.elapsed:.2f}s")



main()

