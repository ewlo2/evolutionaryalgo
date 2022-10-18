from cmath import log
import math
from random import shuffle
import matplotlib.pyplot as plt
import concurrent.futures

def distance(a,b):
    # d = math.sqrt(math.pow(a[0]-a[1], 2) + math.pow(b[0]-b[1], 2))
    d = math.dist(a,b)
    return d

def totalDistance(path, order):
    pathDist = 0
    currPt = None
    for i in order:
        nxtPt = path[i]
        if currPt is not None:
            pathDist += distance(nxtPt, currPt)
        currPt = nxtPt
    pathDist += distance(nxtPt, currPt)
    return pathDist

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
    
def main():
    
    # cities = [(1,4), (3, 5), (10,3), (5, 7)]
    cities = readTxt('tsp.txt')
    print(cities)
    plot, plot_itr = random_search(cities, 10000000)

    plot_fit(plot, plot_itr)
    
    
if __name__=='__main__':
        main()