import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import os



class Punto:
    def __init__(self, nombre, x, y, tipo):
        self.nombre = nombre
        self.x = x
        self.y = y
        self.tipo = tipo

    def distancia(self, otro):
        if self.tipo == "EUC":
            dx = abs(self.x - otro.x)
            dy = abs(self.y - otro.y)
            return np.sqrt(dx**2 + dy**2)
        else:
            pass

    def __repr__(self):
        return f"({self.nombre})"



class Evaluacion:
    def __init__(self, camino):
        self.camino = camino
        self.dist_total = 0
        self.score = 0.0

    def calcularDistancia(self):
        if self.dist_total == 0:
            total = 0
            for i in range(len(self.camino)):
                actual = self.camino[i]
                siguiente = self.camino[(i + 1) % len(self.camino)]
                total += actual.distancia(siguiente)
            self.dist_total = total
        return self.dist_total

    def calcularFitness(self):
        if self.score == 0:
            self.score = 1 / float(self.calcularDistancia())
        return self.score



def generarCamino(listaPuntos):
    return random.sample(listaPuntos, len(listaPuntos))

def poblacionInicial(tamano, listaPuntos):
    return [generarCamino(listaPuntos) for _ in range(tamano)]



def ordenarPoblacion(poblacion):
    resultados = {i: Evaluacion(p).calcularFitness() for i, p in enumerate(poblacion)}
    return sorted(resultados.items(), key=operator.itemgetter(1), reverse=True)



def seleccion(ranking, numElite):
    seleccionados = []
    df = pd.DataFrame(np.array(ranking), columns=["Index", "Fitness"])
    df["cum_sum"] = df.Fitness.cumsum()
    df["cum_perc"] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(numElite):
        seleccionados.append(ranking[i][0])

    for _ in range(len(ranking) - numElite):
        pick = 100 * random.random()
        for i in range(len(ranking)):
            if pick <= df.iat[i, 3]:
                seleccionados.append(ranking[i][0])
                break
    return seleccionados



def crearMatingPool(poblacion, seleccionados):
    return [poblacion[i] for i in seleccionados]



def cruzar(padre, madre):
    hijo = []
    genA, genB = sorted([int(random.random() * len(padre)), int(random.random() * len(padre))])
    segmento = padre[genA:genB]
    hijo = segmento + [c for c in madre if c not in segmento]
    return hijo


def cruzarPoblacion(pool, numElite):
    hijos = []
    longitud = len(pool) - numElite
    mezcla = random.sample(pool, len(pool))

    hijos.extend(pool[:numElite])
    for i in range(longitud):
        hijo = cruzar(mezcla[i], mezcla[-i - 1])
        hijos.append(hijo)
    return hijos



def mutar(individuo, tasa):
    for i in range(len(individuo)):
        if random.random() < tasa:
            j = int(random.random() * len(individuo))
            individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo


def mutarPoblacion(poblacion, tasa):
    return [mutar(ind.copy(), tasa) for ind in poblacion]



def siguienteGeneracion(actual, numElite, tasa):
    ranking = ordenarPoblacion(actual)
    seleccionados = seleccion(ranking, numElite)
    pool = crearMatingPool(actual, seleccionados)
    hijos = cruzarPoblacion(pool, numElite)
    nuevaGen = mutarPoblacion(hijos, tasa)
    return nuevaGen



def algoritmoGenetico(ciudades, N, maxIter, fracElite=0.2, fracCrossover=0.6, fracMutation=0.2):
    early_stopping_counter = 0 #Nos servirá para parar el algoritmo si las soluciones empiezan a parecerse 
    previous_value = 0
    # Validación rápida
    total_frac = fracElite + fracCrossover + fracMutation
    if not np.isclose(total_frac, 1.0):
        raise ValueError("La suma de los fraccionamientos debe ser 1.0")

    
    poblacion = poblacionInicial(N, ciudades)
    progreso = [1 / ordenarPoblacion(poblacion)[0][1]]
    print(f"Distancia inicial: {progreso[0]:.2f}")
    mejores_rutas = [poblacion[ordenarPoblacion(poblacion)[0][0]]]

    
    numElite = int(fracElite * N)
    numCrossover = int(fracCrossover * N)
    numMutation = N - numElite - numCrossover  # por seguridad

    for gen in range(1, maxIter + 1):
        # Ranking
        ranking = ordenarPoblacion(poblacion)
        seleccionados = [i for i, _ in ranking[:numElite]]  # elite
        eliteIndividuos = [poblacion[i] for i in seleccionados]

        # Cruce
        poolCruce = crearMatingPool(poblacion, seleccionados)
        hijos = []
        mezcla = random.sample(poolCruce, len(poolCruce))
        for i in range(numCrossover):
            padre = mezcla[i % len(mezcla)]
            madre = mezcla[-(i % len(mezcla)) - 1]
            hijos.append(cruzar(padre, madre))

        # Mutación
        mutados = mutarPoblacion(hijos[:numMutation], 1.0)  # mutación completa
        nuevosIndividuos = eliteIndividuos + hijos[numMutation:] + mutados
        poblacion = nuevosIndividuos[:N]  # ajustar si hay exceso
        mejor_idx = ordenarPoblacion(poblacion)[0][0]
        mejor = 1 / ordenarPoblacion(poblacion)[0][1]
        progreso.append(mejor)
        mejores_rutas.append(poblacion[mejor_idx])

        if gen % 10 == 0:
            print(f"Generación {gen}: distancia = {mejor:.2f}")

        if gen != 0:
            if mejor - previous_value <= 1e-3 :
                early_stopping_counter +=1
            else:
                early_stopping_counter = 0

            if early_stopping_counter >= 20:
                print(f"Early stopping en la generación: {gen}, el anterior fue {previous_value} y el actual {mejor} ")
                break
        previous_value = mejor


    fig, ax = plt.subplots(figsize=(6, 6))
    def actualizar(i):
        ax.clear()
        camino = mejores_rutas[i]
        x = [p.x for p in camino] + [camino[0].x]
        y = [p.y for p in camino] + [camino[0].y]
        ax.plot(x, y, marker='o', color='tab:blue')
        ax.set_title(f"Generación {i} | Distancia = {progreso[i]:.2f}")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.grid(True)

    anim = animation.FuncAnimation(fig, actualizar, frames=len(mejores_rutas), interval=200, repeat=False)
    plt.close(fig)
    return mejores_rutas[-1], anim
