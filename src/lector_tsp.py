
from alg_genetico_ import Punto

def leer_tsp(file_path, problem_type):
    puntos = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    in_node_section = False
    for line in lines:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            in_node_section = True
            continue
        if line == "EOF":
            break
        if in_node_section:
            partes = line.split()
            if len(partes) >= 3:
                idx = int(partes[0])
                x = float(partes[1])
                y = float(partes[2])
                puntos.append(Punto(nombre=idx, x=x, y=y, tipo = problem_type))
    return puntos
