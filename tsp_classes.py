# import numpy as np
import random
import matplotlib.pyplot as plt


class City(object):
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = ((xDis ** 2) + (yDis ** 2)) ** 0.5
        return distance
    
    def __repr__(self):
        return "({}: {} x, {} y)".format(self.name, self.x, self.y)


class Route(object):
    def __init__(self, cityList=None, route=None):
        if cityList:
            self.cityList = cityList
            self.route = self.createRoute()
        elif route:
            self.route = route
            self.cityList = route
        else:
            raise IOError('No cityList or route given..')

        self.distance = self.getDistance()
        self.fitness = 1 / self.distance
        
    def createRoute(self):
        return random.sample(self.cityList, len(self.cityList))
    
    def getDistance(self):
        pathDistance = 0
        for i, city in enumerate(self.route):
            try:
                nextCity = self.route[i + 1]
            except IndexError:
                nextCity = self.route[0]
            pathDistance += city.distance(nextCity)
        return pathDistance
    
    def plot(self):
        coords = [(city.x, city.y) for city in self.cityList]
        Xs, Ys = zip(*coords)
        plt.plot(Xs, Ys)
        plt.scatter(Xs, Ys)
        plt.show()
    
    def __repr__(self):
        return '{}'.format('.'.join([str(c.name) for c in self.route]))
    
    def __len__(self):
        return len(self.cityList)