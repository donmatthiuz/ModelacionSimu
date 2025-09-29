import random
import math
from typing import List, Tuple

""" tiene una funcion para generar conjunto de ciudades inventadas, y que luego guarde esas coordenadas en un tsp valido para usarlo luego"""

def make_invented_instance(n: int, seed: int = 42) -> List[Tuple[float, float]]:
    """es para generar ciudades inventadas en forma de dos clusters"""
    random.seed(seed)
    coords = []
    c1, c2 = (20, 20), (80, 80)
    for _ in range(n // 2):
        coords.append((random.gauss(c1[0], 8), random.gauss(c1[1], 8)))
    for _ in range(n - n // 2):
        coords.append((random.gauss(c2[0], 8), random.gauss(c2[1], 8)))
    return coords


def save_tsplib(coords: List[Tuple[float, float]], path: str, name="CUSTOM"):
    with open(path, "w") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write("COMMENT: inventado\n")
        f.write(f"DIMENSION: {len(coords)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x:.3f} {y:.3f}\n")
        f.write("EOF\n")

