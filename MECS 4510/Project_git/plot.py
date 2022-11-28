import matplotlib.pyplot as plt
import pickle

def savePickle(filename, list):
    with open(filename, "wb") as fp:
        pickle.dump(list, fp)

def openPickle(filename):
    
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

t = openPickle('time_per_second')
cycle_sec = openPickle('cycles_per_second')
plt.plot(t, cycle_sec, label = 'Bouncing Cube: Spring evaluations per second')
plt.title("Bouncing Cube: Spring Evaluations per Second")
plt.xlabel('T [s]')
plt.ylabel('Spring evaluations per second')
plt.show()
