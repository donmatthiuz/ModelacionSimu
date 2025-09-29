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

