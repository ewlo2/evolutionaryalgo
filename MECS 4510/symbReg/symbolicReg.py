from cProfile import label
from cmath import log
import math
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from random import randint, shuffle, choice, randrange
import matplotlib.pyplot as plt
import numpy as np
import pickle
class Node(object):
    def __init__(self, value = 0, tp = None):
        self.__value = value
        self.__tp = tp
        self.__child = []

    def child(self):
        if not self.__tp == 'terminal':
            return self.__child
    
    def setChild(self, obj):
        if not self.__tp == 'terminal':
            if obj is not None:
                self.__child.append(obj)
   
    def value(self):
        return self.__value
    
    def setValue(self, val):
        if val is not None:
            self.__value = val
    
    def tp(self):
        return self.__tp

    def setTp(self, val):
        if val is not None:
            self.__tp = val
    
    
    
class Funct(object):
    def __init__(self, func, target):
        self.func = func
        self.target = target
        self.mae = None
        
    # def safeDiv(self, a, b):
        
    #     if (b < 0.001) or (b> -0.001):
    #         return 1
    #     else:
    #         return np.divide(a,b)
    
    def safeDiv(self, a, b):
        return np.divide(np.multiply(a,b),(b**2+1e-20))
    def sqRoot(self, a):
        return math.sqrt(abs(a))
    

        
    def operators(self, node, left, right):
        # if self.checkFloat(func):
        #     if func > 10:
        #         func = 10
        #     if func < -10:
        #         func = -10
        #     return func
        if node.value() == 'mul':
                return np.multiply(left, right)
        elif node.value() == 'div':
            return self.safeDiv(left, right)
        elif node.value() == 'add':
            return np.add(left,right)
        elif node.value() == 'sub':
            return np.subtract(left,right)
        if node.value() == 'sine':
            return np.sin(left)
        if node.value() == 'cos':
            return np.cos(left)

        
    def calculate(self, func):
        parent = func
        if parent:
            value = parent.value()
            if parent.tp() == 'terminal':
                if value == 'x':
                    x = np.linspace(-10,10, 1000, endpoint=True)
                    return x
                else:
                    return float(value)
            if parent.child():
                children = parent.child()
                num = len(children)
                leftChild = self.calculate(children[0])
                if num<2:
                    return self.operators(parent, leftChild, None)
                if children[1]:
                    rightChild = self.calculate(children[1])
                    return self.operators(parent, leftChild, rightChild)
    def calculateTree(self):
        y = self.calculate(self.func)
        return y

    def fitness_mae(self):
        test = np.array(self.calculateTree())
        tar = np.array(self.calculate(self.target))
        self.mae = np.mean(np.abs(tar-test))
        return self.mae
            
def checkFloat(str):
        try:
            str = float(str)
            return True
        except ValueError:
            return False
        
def functTree(list):
    dic = {'add': 2, 'sub' : 2, 'div': 2, 'mul':2, 'sine':1, 'cos':1}
    value = list[0]
    if  value in dic.keys():
        node = Node(value, tp = 'function')
        list.pop(0)
        for i in range(dic[value]):
            node.setChild(functTree(list))
        return node
    else:
        node = Node(value, tp = 'terminal')
        list.pop(0)
        return node    

def randomFunc(depth = 3):
    dic = {'add': 2, 'sub' : 2, 'div': 2, 'mul':2, 'sine':1, 'cos':1}
    func = []
    depth = randint(0,depth)
    if depth > 0:
        keys = list(dic)
        key = keys[randint(0,5)]
        func.append(key)
        depth -= 1
        for ad in range(dic[key]):
            func.extend(randomFunc(depth))
        return func
    else:
        terminal = float(randint(-10000,10000))/1000
        term = [str(terminal), 'x']
        return [choice(term)]

def randomSearch(truthNode):
    bestFunc = None
    itr_perf = []
    bestFitness = []
    iter = int(1e6)
    depth = 3
   
    for i in range(iter):
        functRand = randomFunc(depth)
        node = functTree(functRand)
        tree = Funct(node, truthNode)
        if not bestFitness:
            bestFitness.append(tree.fitness_mae())
            itr_perf.append(i)
            bestFunc = tree
            continue
        elif tree.fitness_mae()<bestFitness[-1]:
            bestFitness.append(tree.fitness_mae())
            itr_perf.append(i)
            bestFunc = tree           
    return [bestFitness[-1], bestFunc], bestFitness, itr_perf


def savePickle(filename, list):
    with open(filename, "wb") as fp:
        pickle.dump(list, fp)

def openPickle(filename):
    
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def plotResult(result, original, title):
    x = np.linspace(-10,10, 1000, endpoint=True)
    plt.plot(x, result,'-', label="result")
    plt.plot(x, original, '.', label = "original")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.show()   
    
def plotLearning(fitnessList, itrList):
    plt.plot(itrList, fitnessList,'-', label="result")
    plt.show()
    
if __name__ == '__main__':    
    #Cos(2.5x)*x
    truth = ['mul', 'cos', 'mul','2.5' , 'x', 'x']

    # iterations = 100000
    # pool = mp.Pool(mp.cpu_count())
    
    
    truthNode = functTree(truth)
    truthTree = Funct(truthNode, truthNode)
    
    # results, fitnessList, itrList = randomSearch(truthNode)
    
    # # Save Results as Pickle
    # savePickle("randomFitnessList", fitnessList)
    # savePickle("randomItrList", itrList)
    # savePickle("randFuncObj", results[1])

    # Open Results
    bestNode = openPickle("randFuncObj")
    fitnessList = openPickle("randomFitnessList")
    itrList = openPickle("randomItrList")

    #Plot Dot Curve
    y_random = bestNode.calculateTree()
    y_truth = truthTree.calculateTree()
    plotResult(y_random, y_truth, 'Random Search')
    
    #Plot Learning Curve
    plotLearning(fitnessList, itrList)
    
    print (f'Random Search MAE: {bestNode.fitness_mae()}')
    
    
    # pool.close()

