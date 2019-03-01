import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from copy import deepcopy

import pdb

from tsp_classes import City, Route

def createCities(nCities, center, spread):
    return [
        City(
        x=int(np.random.normal(center, spread)),
        y=int(np.random.normal(center, spread)),
        name=i
        )
        for i in range(nCities)
    ]

def createRoutes(cities, nRoutes):
    return [Route(cities) for i in range(nRoutes)]

def rankRoutes(routes):
    return sorted(routes, key=lambda route: route.fitness, reverse=True)

def selectRoutes(routes, eliteProp=0.2):
    rankedRoutes = rankRoutes(routes)
    eliteCount = int(eliteProp * len(routes))
    poolCount = len(routes) - eliteCount
    elites = rankedRoutes[:eliteCount]  # Auto-winners
    pool = rankedRoutes[eliteCount:-eliteCount]  # pool less auto-losers
    fitness_scores =  [r.fitness for r in pool]
    sum_fitness_scores = sum(fitness_scores)  # Use normalized [0-1] fitness as probability
    fitness_probs = [fitness_score / sum_fitness_scores for fitness_score in fitness_scores]
    return elites, [np.random.choice(pool, p=fitness_probs) for _ in range(poolCount)]

def combineRoute(route1, route2):
    routeLength = len(route1)
    geneA = int(random.random() * routeLength)
    geneB = int(random.random() * routeLength)
    startGene, endGene = min(geneA, geneB), max(geneA, geneB) - 1
    insertion_point = int(random.random() * routeLength)
    route1_segment = route1.route[startGene:endGene]
    route2_front = [c for c in route2.route[:insertion_point] if c not in route1_segment]
    route2_back  = [c for c in route2.route[insertion_point:] if c not in route1_segment]
    combined = route2_front + route1_segment + route2_back
    return Route(route=combined)

def combineRoutes(routes):
    n_routes = len(routes)
    pairs = []
    new_routes = []
    while len(pairs) < n_routes:
        r1, r2 = np.random.choice(routes, size=2, replace=False)
        if r1.route != r2.route:
            pairs.append((r1, r2))
    for (r1, r2) in pairs:
        new_routes.append(combineRoute(r1, r2))
    return new_routes

def mutateRoute(route, mutationRate=0.05):
    for ix, _ in enumerate(route.route):
        if random.random() < mutationRate:
            dst = int(random.random() * len(route.route))
            route.route[ix], route.route[dst] = route.route[dst], route.route[ix]
    return route

def mutateRoutes(routes, mutationRate=0.05):
    return [mutateRoute(route, mutationRate) for route in routes]

def iterGeneration(currentRoutes, eliteProp, mutationRate):
    rankedRoutes = rankRoutes(currentRoutes)
    elites, pool = selectRoutes(rankedRoutes, eliteProp)
    children = combineRoutes(pool)
    results = {
        'elites': elites,
        'pool': mutateRoutes(children, mutationRate),
        'bestRoute': elites[0],
        'meanEliteDistance': np.mean([route.distance for route in elites]),
        'meanPoolDistance': np.mean([route.distance for route in pool]),
        }
    return results

def geneticAlgorithm(nCities, nRoutes, eliteProp, mutationRate, generations, plot=True, verbose=True):
    cities = createCities(nCities, center=100, spread=10)
    routes = createRoutes(cities, nRoutes)

    bestStartingRoute = rankRoutes(routes)[0]
    startXs = [city.x for city in bestStartingRoute.route]
    startYs = [city.y for city in bestStartingRoute.route]

    evolution_results = []

    for _ in range(generations):
        results = iterGeneration(routes, eliteProp, mutationRate)
        evolution_results.append(results)
        routes = results['elites'] + results['pool']
    
    best_routes = [result['bestRoute'].distance for result in evolution_results]
    meanEliteDistance = [result['meanEliteDistance'] for result in evolution_results]
    meanPoolDistance = [result['meanPoolDistance'] for result in evolution_results]
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
        ax1.plot(best_routes, label='Best Route')
        ax1.plot(meanEliteDistance, label='Mean Elite Route')
        ax1.plot(meanPoolDistance, label='Mean Pool Route')

        ax2.scatter(startXs, startYs)
        ax2.plot(startXs + [startXs[0]], startYs + [startYs[0]], label='Starting')

        finalXs = [city.x for city in routes[0].route]
        finalYs = [city.y for city in routes[0].route]
        ax3.scatter(finalXs, finalYs)
        ax3.plot(finalXs + [finalXs[0]], finalYs + [finalYs[0]], label='Ending')

        ax1.set_ylabel('Distance')
        ax1.set_xlabel('Generations')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax3.set_xlabel('X'); ax3.set_ylabel('Y')
        ax1.legend(); ax2.legend(); ax3.legend()
        plt.show()
        
    return routes

if __name__ == '__main__':
    
    ga = geneticAlgorithm(
        nCities=10,
        nRoutes=20,
        eliteProp=0.1,
        mutationRate=0.1,
        generations=1000,
        plot=True
        )

    # ga[0].plot()