from pathlib import Path
from typing import List, Tuple
import math

class TSPLIBInstance:
    """Guarda coordenadas y calcula distancias según tipo (EUC_2D o GEO)"""
    def __init__(self, name: str, coords: List[Tuple[float, float]], weight_type: str):
        self.name = name
        self.coords = coords
        self.n = len(coords)
        self.weight_type = weight_type.upper()

    def dist(self, i: int, j: int) -> float:
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]

        if self.weight_type == "EUC_2D":
            return math.hypot(xi - xj, yi - yj)
        elif self.weight_type == "GEO":
            return self.geo_distance(xi, yi, xj, yj)
        else:
            raise ValueError(f"EDGE_WEIGHT_TYPE {self.weight_type} no soportado")

    def geo_distance(self, lat1, lon1, lat2, lon2):
        """Calcula la distancia GEO según la fórmula de TSPLIB"""
        # Convertir grados.minutos a radianes
        def to_radians(x):
            deg = int(x)
            min_ = x - deg
            return math.pi * (deg + 5.0 * min_ / 3.0) / 180.0

        RRR = 6378.388  # Radio de la Tierra según TSPLIB
        lat1, lon1, lat2, lon2 = map(to_radians, [lat1, lon1, lat2, lon2])

        q1 = math.cos(lon1 - lon2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)

        return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    def distance_matrix(self):
        D = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    D[i][j] = self.dist(i, j)
        return D


def load_tsplib_2d(path: str | Path) -> TSPLIBInstance:
    """Lee archivo .tsp y detecta EDGE_WEIGHT_TYPE"""
    path = Path(path)
    name = path.stem
    coords = []
    weight_type = "EUC_2D"

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    for ln in lines:
        if ln.upper().startswith("EDGE_WEIGHT_TYPE"):
            weight_type = ln.split(":")[1].strip().upper()
        if ln.upper().startswith("NODE_COORD_SECTION"):
            break

    start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("NODE_COORD_SECTION")) + 1
    for ln in lines[start:]:
        if ln.upper().startswith("EOF"):
            break
        parts = ln.split()
        if len(parts) >= 3:
            _, x, y = parts[:3]
            coords.append((float(x), float(y)))

    return TSPLIBInstance(name, coords, weight_type)
