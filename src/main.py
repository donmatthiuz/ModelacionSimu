import argparse
from pathlib import Path
from tsplib import load_tsplib_2d
from tsp_utils import make_invented_instance, save_tsplib
from lp_tsp import solve_tsp_mtz
import matplotlib.pyplot as plt


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

def plot_tsp_route(coords, route, title="TSP Route", filename=None):
    plt.figure(figsize=(12, 10))
    
    # Coordenadas del tour
    x_coords = [coords[i][0] for i in route] + [coords[route[0]][0]]
    y_coords = [coords[i][1] for i in route] + [coords[route[0]][1]]
    
    # Dibujar ruta
    plt.plot(x_coords, y_coords, 'b-', linewidth=1.5, alpha=0.7, label='Ruta')
    
    # Ciudades
    plt.scatter([coords[i][0] for i in route], 
                [coords[i][1] for i in route], 
                c='red', s=50, zorder=5, label='Ciudades')
    
    # Ciudad inicial
    plt.scatter(coords[route[0]][0], coords[route[0]][1], 
                c='green', s=200, marker='*', zorder=6, 
                label=f'Inicio (ciudad {route[0]})')
    
    plt.xlabel('Coordenada X', fontsize=12)
    plt.ylabel('Coordenada Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if filename:
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {filename}")
    
    plt.show()
    
#def main():
#    ap = argparse.ArgumentParser()
#    ap.add_argument("--eil", default="data/eil101.tsp")
#    ap.add_argument("--inventado-n", type=int, default=80)
#    args = ap.parse_args()
    
#     # LP inventado con el tsp utils 
#     inv_coords = make_invented_instance(args.inventado_n, seed=33)
#     save_tsplib(inv_coords, "data/custom_inventado.tsp", name="INVENTADO")
#     Dinv = distance_matrix(inv_coords)
#     lp_res2 = solve_tsp_mtz(Dinv, time_limit=300)
#     print(f"[inventado] Status={lp_res2.status}, Distancia={lp_res2.length:.2f}, Tiempo={lp_res2.elapsed:.2f}s")

    # LP en eil101 
#    eil = load_tsplib_2d(args.eil)
#    D = distance_matrix(eil.coords)
#    lp_res = solve_tsp_mtz(D)
#    print(f"[eil101] Status={lp_res.status}, Distancia={lp_res.length:.2f}, Tiempo={lp_res.elapsed:.2f}s")



#main()

