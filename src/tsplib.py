from pathlib import Path
from typing import List, Tuple
import math

"""
Este py sirve para leer y parsear archivos tsp
"""

class TSPLIBInstance:
    """ Esta instancia guarda coordenadas de las ciudades, y luego calcula las distancias"""
    def __init__(self, name: str, coords: List[Tuple[float, float]]):
        self.name = name
        self.coords = coords
        self.n = len(coords)

    def dist(self, i: int, j: int) -> float:
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        return math.hypot(xi - xj, yi - yj)

    def distance_matrix(self):
        D = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    D[i][j] = self.dist(i, j)
        return D


def load_tsplib_2d(path: str | Path) -> TSPLIBInstance:
    """Lee archivo .tsp """
    path = Path(path)
    name = path.stem
    coords = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("NODE_COORD_SECTION")) + 1
    for ln in lines[start:]:
        if ln.upper().startswith("EOF"):
            break
        parts = ln.split()
        if len(parts) >= 3:
            _, x, y = parts[:3]
            coords.append((float(x), float(y)))

    return TSPLIBInstance(name, coords)

