from ast import Or
from cProfile import label
from cmath import log
import collections
import math
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from random import randint, shuffle, choice, randrange, sample
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.setrecursionlimit(1500)


class Node(object):
    def __init__(self, value = 0, tp = None):
        self.__value = value
        self.__tp = tp
        self.__child = []

    def child(self):
        if not self.__tp == 'terminal':
            return self.__child
        else:
            return None
    
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
            
    def display(self):
        lines, *_ = self.showTree()
        for line in lines:
            print(line)
    
    def showTree(self):
        
        children = self.child()
        if not children:
            line = '%s' % self.value()
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if len(children) == 1:
            lines, n, p, x = children[0].showTree()
            s = '%s' % self.value()
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        # if len(children) == 2:
        #     lines, n, p, x = children[1].showTree()
        #     s = '%s' % self.value()
        #     u = len(s)
        #     first_line = s + x * '_' + (n - x) * ' '
        #     second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
        #     shifted_lines = [u * ' ' + line for line in lines]
        #     return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = children[0].showTree()
        right, m, q, y = children[1].showTree()
        s = '%s' % self.value()
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2
    
    
    
class Funct(object):
    def __init__(self, func, target = None):
        self.func = func
        self.target = target
        self.mae = None
            
    def safeDiv(self, a, b):
        return np.divide(np.multiply(a,b),(b**2+1e-20))
    
    def sqRoot(self, a):
        return math.sqrt(abs(a))
    
    def func(self):
        return self.func
    
    def setFunc(self, func):
        self.func = func
    
    def getChildLevel(self):        
        child = self.func.child()
        return child
            
    def operators(self, node, left, right):
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
        else:
            return np.zeros(1000)
        
    def calculateTree(self):
        y = self.calculate(self.func)
        return y

    def fitness_mae(self):
        test = np.array(self.calculateTree())
        tar = np.array(self.calculate(self.target))
        self.mae = np.mean(np.abs(tar-test))
        return self.mae
    
    def addTest(self, tar):
        if tar:
            self.target = tar
            self.fitness_mae()
        return self.mae
    
    def display(self):
        return self.func.display()
        
            
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
    iter = int(1e7)
    depth = 4
   
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
    if (not itr_perf[-1] == i-1):
        bestFitness.append(bestFitness[-1])
        itr_perf.append(i-1)
    return [bestFitness[-1], bestFunc], bestFitness, itr_perf

def initPopulation(depth, popSize = 1000): 
    popFunc= [randomFunc(depth) for i in range(popSize)]
    nodeList = [functTree(node) for node in popFunc]
    tree = [Funct(x) for x in nodeList]
    return tree

def calculateGAFit(test, pop):
    popFit = []  
    
    for func in pop:
        fit = func.addTest(test)
        popFit.append([fit, func])
    popFit= np.array(popFit)
    
    return popFit

def selectionTour(popFit, tourSize, depth, pCrossover = True):
    if pCrossover:
        idx = range(len(popFit))
        idx1 = sample(idx, tourSize)
        idx2 = sample(idx, tourSize)
        
        tour1 = popFit[idx1]
        tour2 = popFit[idx2]
        pop1 = tour1[tour1[:,0].argsort()][0]
        pop2 = tour2[tour2[:,0].argsort()][0]
        return crossover(pop1, pop2, depth)
    else:
        tour = sample(popFit, tourSize)
        pop = tour[tour[:,0].argsort()][0]
        return pop
    
def crossover(pop1, pop2, depth, pCross = 0.3):
    node1 =  pop1[1].func
    node2 =  pop2[1].func
    orgTree1= [[0,node1]]
    orgTree2 = [[0,node2]]
    
    funcNode1 = node1
    funcNode2 = node2
    lvl = randrange(1, depth)
    if (randint(0,100)/100 <= pCross):
        for i in range(lvl):
            if funcNode1.child() and funcNode2.child():
                funcNode1 = funcNode1.child()
                fN1 = randrange(len(funcNode1))
                funcNode1 = funcNode1[fN1]
                orgTree1.append([fN1, funcNode1])
                
                funcNode2 = funcNode2.child()
                fN2 = randrange(len(funcNode2))
                funcNode2 = funcNode2[fN2]
                orgTree2.append([fN2, funcNode2])
            else:
                break
        lvl = i
        if len(orgTree1)>1:
            side1 = orgTree1[-1][0]
            parent = orgTree1[-2][1]
            child = orgTree2[-1][1]
            
            parent.child()[side1] = child
            
            for i in range(lvl-1):   
                side = orgTree1[-(i+2)][0]
                child = orgTree1[-(i+2)][1]
                parent = orgTree1[-(i+3)][1]
                parent.child()[side] = child
            cross = Funct(parent)
        else:
            cross = Funct(orgTree1[-1][1])
    else:
        return pop1[1]
    # print("Lvl:" + str(lvl))
    # print('Node 1:')
    # pop1[1].func.display()
    # print('Node 2:')
    # pop2[1].func.display()
    # print('Cross:')
    
    # cross.display()
    
    return cross      

def mutate(child, depth, pMutate = 0.3):
    dic = {'add': 2, 'sub' : 2, 'div': 2, 'mul':2, 'sine':1, 'cos':1}
    tree = child
    p = float(randint(0,10))/10
    if ( p <= pMutate):
        d = randint(1,depth)
        child = child.func
        orgTree = [[0, child]]
        for i in range(d):
            if child.child():
                dside = randrange(len(child.child()))
                child = child.child()[dside]
                orgTree.append([dside,child])
            else:
                break
        if child.tp() == 'function':
            value = choice(list(dic.keys()))
            while not(dic[child.value()] == dic[value]):
                value = choice(list(dic.keys()))
            child.setValue(value)
        if child.tp() == 'terminal':
            terminal = float(randint(-10000,10000))/1000
            term = [str(terminal), 'x']
            value = choice(term)
            child.setValue(value)
        # orgTree[-1][1] = child
        # for i in range(len(orgTree)-1):
        #    parent = orgTree[-(i+2)][1]
        #    childNode = orgTree[-(i+1)][1]
        #    side = orgTree[-(i+1)][0]
        #    parent.child()[side] = childNode
        return(tree)
    else:       
        return(child)

def GA(test, popSize = 1000, selecTour = True ):
    depth = 4
    n_itr = int(1e3)
    bestFunc = None
    itr_perf = []
    bestFitness = []
    test = test
    
    popTree = initPopulation(depth, popSize)
    popFit = calculateGAFit(test, popTree)
    bestFunc = popFit[popFit[:,0].argsort()][0][1]
    bestFitness.append(popFit[popFit[:,0].argsort()][0][0])
    itr_perf.append(1)   
    for i in range(n_itr):
        if selecTour:
            tourSize = 50
            children = []
            for i in range(popSize):
                pop = selectionTour(popFit,tourSize, depth)
                children.append(pop)

        mutated = [mutate(x, depth) for x in children]
        popFit = calculateGAFit(test, mutated)
        bestFunc = popFit[popFit[:,0].argsort()][0][1]
        bestFitness.append(popFit[popFit[:,0].argsort()][0][0])
        itr_perf.append(i+1)   
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
    
    # # Random Search
    # results, fitnessList, itrList = randomSearch(truthNode)
    
    # GA
    results, fitnessList, itrList = GA(truthNode)

    # # Save Results as Pickle
    # savePickle("randomFitnessList", fitnessList)
    # savePickle("randomItrList", itrList)
    # savePickle("randFuncObj", results[1])
    
    savePickle("GAFitnessList", fitnessList)
    savePickle("GAItrList", itrList)
    savePickle("GAFuncObj", results[1])

    # # Open Results
    # bestNode = openPickle("symbReg\\randomSearch\depth4\\10e7\\randFuncObj")
    # fitnessList = openPickle("symbReg\\randomSearch\depth4\\10e7\\randomFitnessList")
    # itrList = openPickle("symbReg\\randomSearch\depth4\\10e7\\randomItrList")
    
    # bestNode = openPickle("GAFuncObj")
    # fitnessList = openPickle("GAFitnessList")
    # itrList = openPickle("GAItrList")


    # #Plot Dot Curve
    # y_random = bestNode.calculateTree()
    # y_truth = truthTree.calculateTree()
    # plotResult(y_random, y_truth, 'Random Search')
    
    # #Plot Learning Curve
    # plotLearning(fitnessList, itrList)
    
    # print (f'Random Search MAE: {bestNode.fitness_mae()}')
    
    
    # pool.close()

