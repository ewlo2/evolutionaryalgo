from cmath import log
import math
from random import randint, shuffle, choice, randrange
import matplotlib.pyplot as plt
import operator

class City:
    def __init__(self,cord):
        self.cord = cord
    
class Route:
    def __init__(self, route):
         self.route = route
         self.distance = 0
         self.fitness = 0
         
    def routeDistance(self):
        pathDist = 0
        for i in range(len(self.route)-1):
            currPt = self.route[i].cord()
            nxtPt = self.route[i+1].cord()
            pathDist += distance(nxtPt, currPt)
        pathDist += distance(nxtPt, self.route[0].cord())
        self.distance = pathDist
        return self.distance           
    
    def routeFitness(self):
        self.fitness = 1/float(self.routeDistance())
        return self.fitness
    
def distance(a,b):
    # d = math.sqrt(math.pow(a[0]-a[1], 2) + math.pow(b[0]-b[1], 2))
    d = math.dist(a,b)
    return d

# def totalDistance(path, order):
#     pathDist = 0
#     currPt = None
#     for i in order:
#         nxtPt = path[i]
#         if currPt is not None:
#             pathDist += distance(nxtPt, currPt)
#         currPt = nxtPt
#     pathDist += distance(nxtPt, currPt)
#     return pathDist

def initialPopulation(cities,popSize = 100):
    order = [i for i in range(len(cities))]
    cities_ordered = []
    population = []
    
    for i in range(popSize):
        shuffle(order)
        for i in range(0, len(order)):
            cities_ordered.append(City(cities[order[i]]))
        population.append(cities)
    return population

def allFitness(pop):
    fit = {}
    for i in pop:
        fit[i] = Route(i).routeFitness()
    return fit

def ranking(pop):
    pop = allFitness(pop)
    return sorted(pop.items(), key = pop.get, reverse = True)

def selection(pop):
    fitness = allFitness(pop)
    totalFit = sum(fitness.values())
    probability = list(fitness.values())/totalFit
    idx = list(range(len(pop)))
    idx_A = choice(idx, len(pop), p = probability, replace = True)
    idx_B = choice(idx, len(pop), p = probability, replace = True)

    parentA = pop[idx_A]
    parentB = pop[idx_B]
    
    return  parentA, parentB

def crossover(parentA, parentB):
    offspring = []
    
    idxA = randrange(len(parentA))
    idxB = randrange(len(parentB))
    start = min(idxA, idxB)
    end = max(idxA,idxB)
    offspring = parentA[start:end] 
    
    for i in parentB:
        if not i in offspring:
            offspring.append(i)
            
    return offspring
        
def mutate(route, mutationRate= 0.3):
    for i in range(int(len(route)*mutationRate)):
        cityA = randint(0,range(len(route)))
        cityB = randint(0, range(len(route)))

        route[cityA], route[cityB] = route[cityB], route[cityA]
    return route

def mutation(pop):
    pop = []
    for i in pop:
        pop.append(mutate(i))
    return pop
    
def GA(pop, mutate = True, crossover = True):
    if crossover:
        selectA, selectB = selection(pop)
        pop = crossover(selectA, selectB)
    elif not crossover:
        rank = ranking(pop)
        pop = mutation(pop)
        

def random_search(cities, nr_iterations): 
    best_order = None
    best_path_length = None
    plot = []
    plot_itr = []
    order = [i for i in range(len(cities))]
    # print(order)
    for i in range(nr_iterations):
        shuffle(order)
        path_length = totalDistance(cities, order)
        # print(path_length)
        if best_order is None:
                best_order = order
                best_path_length = path_length
                continue
        elif best_path_length > path_length:
                best_order = order
                best_path_length = path_length
        print(best_path_length)
        if not bool(plot):
            plot.append(best_path_length)
            plot_itr.append(i)
            continue
        elif (plot[-1] != best_path_length):
            plot.append(best_path_length)
            plot_itr.append(i)
    return plot, plot_itr
    # print(f"Shortest path was {best_path_length} long")
    # print(best_order)
    
def readTxt(filename):
    cities = []
    with open(filename) as f:
        for lines in f:
            field = lines.strip().split(',')
            cord = (float(field[0]), float(field[1]))
            # print(cord)
            cities.append(cord)
    return cities

def plot_fit(data, n_itr):
    fig1 =  plt.figure()
    ax = fig1.add_subplot(2, 1, 1)
    line, = ax.plot(n_itr, data)
    
    ax.set_xscale('log')
    
    plt.show()
    
def saveTxt(values):
    with open("randomSr10e6.txt", "w") as output:
        for row in values:
            output.write(" ".join(str(row)) + "\n")
            
def main():
    
    # cities = [(1,4), (3, 5), (10,3), (5, 7)]
    cities = readTxt('tsp.txt')
    print(cities)
    plot, plot_itr = random_search(cities, 1000)
    saveTxt([plot, plot_itr])
    plot_fit(plot, plot_itr)
    
    
if __name__=='__main__':
        main()