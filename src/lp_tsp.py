import pulp as pl
import time
from typing import List

class LPTSPResult:
    def __init__(self, route, length, elapsed, status):
        self.route = route
        self.length = length
        self.elapsed = elapsed
        self.status = status


def solve_tsp_mtz(D: List[List[float]], time_limit=None) -> LPTSPResult:
    n = len(D)
    nodes = range(n)
    model = pl.LpProblem("TSP_MTZ", pl.LpMinimize)

    x = pl.LpVariable.dicts("x", (nodes, nodes), 0, 1, pl.LpBinary)
    u = pl.LpVariable.dicts("u", nodes, 1, n, pl.LpContinuous)

    model += pl.lpSum(D[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

    for i in nodes:
        model += pl.lpSum(x[i][j] for j in nodes if j != i) == 1
    for j in nodes:
        model += pl.lpSum(x[i][j] for i in nodes if i != j) == 1

    model += u[0] == 1
    for i in nodes:
        if i == 0: continue
        for j in nodes:
            if j == 0 or i == j: continue
            model += u[i] - u[j] + n * x[i][j] <= n - 1

    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    
    
    t0 = time.perf_counter()
    model.solve(solver)
    elapsed = time.perf_counter() - t0
    status = pl.LpStatus[model.status]

    succ = [-1] * n
    for i in nodes:
        for j in nodes:
            if i != j and pl.value(x[i][j]) > 0.5:
                succ[i] = j
                break

    route, cur = [0], 0
    for _ in range(n - 1):
        cur = succ[cur]
        route.append(cur)

    length = sum(D[route[i]][route[(i + 1) % n]] for i in range(n))
    return LPTSPResult(route, length, elapsed, status)


def solve_tsp_cutplane(D: List[List[float]], time_limit=None, max_iterations=100) -> LPTSPResult:
    n = len(D)
    nodes = range(n)
    
    model = pl.LpProblem("TSP_CutPlane", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", (nodes, nodes), 0, 1, pl.LpBinary)
    
    # Función objetivo
    model += pl.lpSum(D[i][j] * x[i][j] for i in nodes for j in nodes if i != j)
    
    # Restricciones de grado
    for i in nodes:
        model += pl.lpSum(x[i][j] for j in nodes if j != i) == 1
    for j in nodes:
        model += pl.lpSum(x[i][j] for i in nodes if i != j) == 1
    
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    t0 = time.perf_counter()
    
    for iteration in range(max_iterations):
        if time_limit and (time.perf_counter() - t0) > time_limit:
            break
            
        model.solve(solver)
        
        if model.status != pl.LpStatusOptimal:
            break
        
        # Encontrar subtours
        visited = [False] * n
        subtours = []
        
        for start in nodes:
            if visited[start]:
                continue
            
            tour = [start]
            visited[start] = True
            current = start
            
            while True:
                next_city = None
                for j in nodes:
                    if j != current and x[current][j].varValue and x[current][j].varValue > 0.5:
                        next_city = j
                        break
                
                if next_city is None or next_city == start:
                    break
                
                tour.append(next_city)
                visited[next_city] = True
                current = next_city
            
            if len(tour) > 1:
                subtours.append(tour)
        
        # Si solo hay un tour, terminamos
        if len(subtours) == 1:
            break
        
        # Añadir restricciones de eliminación de subtours
        for tour in subtours:
            if len(tour) < n:
                model += pl.lpSum(x[i][j] for i in tour for j in tour if i != j) <= len(tour) - 1
        
        print(f"Iteración {iteration + 1}: {len(subtours)} subtours encontrados")
    
    elapsed = time.perf_counter() - t0
    status = pl.LpStatus[model.status]
    
    # Extraer ruta final
    succ = [-1] * n
    for i in nodes:
        for j in nodes:
            if i != j and x[i][j].varValue and x[i][j].varValue > 0.5:
                succ[i] = j
                break
    
    route, cur = [0], 0
    for _ in range(n - 1):
        if succ[cur] == -1:
            break
        cur = succ[cur]
        route.append(cur)
    
    length = sum(D[route[i]][route[(i + 1) % n]] for i in range(len(route)))
    
    return LPTSPResult(route, length, elapsed, status)



