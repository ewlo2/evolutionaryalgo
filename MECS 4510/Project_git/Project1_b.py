import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from vpython import *
import concurrent.futures as cf
import time
import math
from random import randint, sample, randrange
import pickle
import sim_lib

class Mass(object):
    def __init__(self, m, p, v = np.array([0,0,0]), a = np.array([0,0,0]), f = np.array([0,0,0]), cube_obj = None):
        self.m = m
        self.p = p
        self.p_i = p
        self.v = v
        self.v_i = v
        self.a = a
        self.a_i = a
        self.f = f
        self.f_i = f
        self.cube_obj = cube_obj
        self.gp_e = 0
        self.fl_pe =0
        
    def ma(self):
        return self.m
    def pos(self):
        return self.p
    def set_pos(self, value):
        self.p = value 
    def force(self):
        return self.f
    def set_force(self, value):
        self.f = value   
    def acc(self):
        return self.a
    def set_acc(self, value):
        self.a = value
    def vel(self):
        return self.v
    def set_vel(self, value):
        self.v = value
        
    def cube(self):
        return self.cube_obj
    def set_cube(self, value):
        self.cube_obj = value
        
    def ke(self):
        k_e = 0.5*self.ma()*np.linalg.norm(self.v)**2
        return k_e
    
    def gpe(self):
        self.gp_e = self.ma()*9.81*self.pos()[2]
        return self.gp_e
    
    def flpe(self):
        return self.fl_pe

class Spring(object):
    def __init__(self, L0, m1_index, m2_index, m1 = 0, m2 = 0, k=0):
        self.k = k
        self.L0 = L0
        self.m1 = m1 
        self.m2 = m2 
        self.m1_index = m1_index
        self.m2_index = m2_index
        self.c = 0
        self.a = self.L0
        self.b = 0
        self.w = 10
    def spring_const(self):
        return self.k
    def length(self):
        return np.linalg.norm(self.m2.pos()-self.m1.pos())
    
    def generate_L0(self):
        self.L0 = np.linalg.norm(self.m2.pos()-self.m1.pos())
        self.a = self.L0
    def get_L0(self, value = None):
        if value:
            return self.a + self.b*math.sin(self.w*value + self.c)
        else:
            return self.L0
        
    def set_k(self, value):
        self.k = value    
    def set_c(self, value):
        self.c = value        
    def set_b(self, value):
        self.b = value          
       
    def len_v(self):
        return self.m2.pos()-self.m1.pos()
    def pe(self):
        pe = 0.5*self.spring_const()*np.linalg.norm(self.length()-self.L0)**2
        return pe

class Cube(object):
    def __init__(self, mass, spring, c_m0):
        self.mass = mass
        self.spring = spring
        self.c_m0 = c_m0
        self.c_m = self.c_m0
    def get_pos(self):
        pos = []
        for i in self.mass:
            pos.append(i.pos())
        return pos
    def cal_cm(self):
        pos = np.array(self.get_pos())
        x = np.average(pos[:,0])
        y = np.average(pos[:,1])
        z = np.average(pos[:,2])
        self.c_m = [x, y, z]
    def get_cm(self):
        return self.c_m
    
class Body(object):
    def __init__(self, mass, spring, cube):
        self.mass = mass
        self.spring = spring
        self.cube = cube
        self.c_m0 = []
        self.c_m = []
        self.fit = None
    def get_cm0(self):
        c_v = np.zeros(3)
        for c in self.cube:
            c_v=c_v+c.c_m0
        self.c_m0 = c_v/len(self.cube)
    def get_cm(self):
        c_v = np.zeros(3)
        for c in self.cube:
            c.cal_cm()
            c_v += c_v+c.get_cm()
        self.c_m = c_v/len(self.cube)
    def fitness(self):
        self.fit = 1/abs(self.c_m0.tolist()[1]-self.c_m.tolist()[1])
        
# def interaction(spring_list, mass_list, T):
#     kc = 5000
#     mu = 0.25
#     for spring in spring_list:
#         extension_f = spring.spring_const()*(spring.length()-spring.get_L0(T))
#         dir = spring.len_v()/np.linalg.norm(spring.len_v())
#         extension_f = extension_f*dir
#         spring.m1.set_force(spring.m1.force()+extension_f)
#         spring.m2.set_force(spring.m2.force()-extension_f)

#     for mass in mass_list:
#         if mass.pos()[2]<=0:
#             F_p = mass.force()
#             if mass.pos()[2]<0:
#                 F_p = F_p+np.array([0,0,-kc*mass.pos()[2]])
#             F_n = mass.ma()*np.array([0,0,9.81])
#             if np.linalg.norm(F_p)<np.linalg.norm(F_n)*mu:
#                 F_p[0] = 0
#                 F_p[1] = 0
#             if np.linalg.norm(F_p)>=np.linalg.norm(F_n)*mu:
#                 F_p[0] -= np.linalg.norm(F_n)*mu
#                 F_p[1] -= np.linalg.norm(F_n)*mu
#             mass.set_force(F_p)
#             # print(mass.pos()[2])
#             mass.fl_pe = 0.5*kc*(mass.pos()[2])**2
                
#         else:
#             mass.set_force(mass.force()+mass.ma()*np.array([0,0,-9.81]))
        
    
# def intergration(mass_list, dt):
#     damp = 0.999
#     # damp = 1
#     for mass in mass_list:
#         mass.set_acc(mass.force()/mass.ma())
#         mass.set_vel((mass.vel()+mass.acc()*dt)*damp)
#         mass.set_pos(mass.pos()+mass.vel()*dt)
#         mass.set_force([0,0,0])

# def simulation(mass, spring, T, dt):
#     interaction(spring, mass, T)
#     intergration(mass, dt)


def init_cube(length, mass_corner, n):
    
    cm = []
    coord = []
    d = length/2
    st = 0
    for i in range(n):
        cm.append(np.array([d, d, d])+np.array([0, length*i, st*length]))

        x = cm[i][0]
        y = cm[i][1]
        z = cm[i][2]
        a = np.array([
            [x-d, y-d, z-d],
            [x+d, y-d, z-d],
            [x-d, y-d, z+d],
            [x+d, y-d, z+d],
            
            [x-d, y+d, z-d],
            [x+d, y+d, z-d],
            [x-d, y+d, z+d],
            [x+d, y+d, z+d]])
        coord.append(np.round(a, decimals = 2))
    
    
    mass = []
    p = []
    
    for c in coord:
        for point in c:
            if not point.tolist() in p:
                mass.append(Mass(mass_corner, point))
                p.append(point.tolist())
    p = np.array(p)
    spring = []
    # k = 7500
    cubes = []
    for i in range(n):
        mass_cube = []
        spring_cube = []
        for j in range(i*4+1,i*4+8):
            for t in range(j,i*4+8):
                spring.append(Spring(0, j-1, t, mass[j-1],mass[t]))
                spring_cube.append(spring[-1])
            mass_cube.append(mass[j-1])
        mass_cube.append(mass[j])
        cubes.append(Cube(mass_cube, spring_cube, cm[i]))
    body = Body(mass, spring, cubes)
    return body

def init(l, m, n, pop_n):
    population = []
    for _ in range(pop_n):
        body = init_cube(l,m,n)
        population.append(body)
    return population

def init_spring(body, init, k = 7500, c =0, b=0):
    spring = body.spring
    for sp in spring:
        if init:
            k = randint(5000, 7500)
            c = sample([0,314],1)[0]
            b = float(randint(0,10))/100
        sp.set_k(k)
        sp.set_c(c)
        sp.set_b(b)
        sp.generate_L0()

def reset_pos(body):
    body.c_m = body.c_m0
    for i in body.cube:
        i.c_m = i.c_m0
    for j in body.mass:
        j.p = j.p_i
        j.v = j.v_i
        j.a = j.a_i
        j.f = j.f_i
    
# def run(body, T, dt, osc):
#     body.get_cm0()
#     for _ in range(10):
#         T += dt
#         if not osc:
#             simulation(body.mass, body.spring, None, dt)
#         else:
#             simulation(body.mass, body.spring, T, dt)
#     body.get_cm()
#     body.fitness()

def selectionTour(popFit, tourSize,  pCrossover = True):
    popFit = np.array(popFit)
    if pCrossover:
        idx = range(len(popFit))
        idx1 = sample(idx, tourSize)
        idx2 = sample(idx, tourSize)
        
        tour1 = popFit[idx1]
        tour2 = popFit[idx2]
        pop1 = tour1[tour1[:,0].argsort()][0]
        pop2 = tour2[tour2[:,0].argsort()][0]
        return crossover(pop1, pop2)
    else:
        tour = sample(popFit, tourSize)
        body = tour[tour[:,0].argsort()][0][1]
        return body
    
def crossover(body1, body2, pCross = 0.3):
    if (randint(0,100)/100 <= pCross):
        # cu = randrange(0,len(body1[1].cube))
        b2 = body2[1].cube[1]
        b1 = body1[1].cube[1]
        for i in range(len(b2.spring)):
            b1.spring[i].set_k(b2.spring[i].k)
            b1.spring[i].set_b(b2.spring[i].b)
            b1.spring[i].set_c(b2.spring[i].c)
        reset_pos(body1[1])
        return body1[1]          
    elif(body1[0]<=body2[0]):
        reset_pos(body1[1])
        return body1[1]
    elif(body1[0]>body2[0]):
        reset_pos(body2[1])
        return body2[1]

def mutate(body, pMutate = 0.3):
    if (randint(0,100)/100 <= pMutate):
        k_i = randrange(0,len(body.spring))
        body.spring[k_i].set_k(randint(5000, 7500))
        b_i = randrange(0,len(body.spring))
        body.spring[b_i].set_c(sample([0,314],1)[0])
        c_i = randrange(0,len(body.spring))
        body.spring[c_i].set_c(float(randint(0,10))/100)
        reset_pos(body)
        return body
    else:
        return body
    
def GA(T, dt, l, m, n):
    osc = True
    tourSize = 50
    itr = int(1e1)
    pop_n = 1000
    pop = init(l,m,n,pop_n)
    fit_list = []
    for body in pop:
        init_spring(body, True)
        sim_lib.run(body, T, dt, osc)                    
        fit_list.append([body.fit,body])
    for o in range (itr):
        batch = 10
        for p in range(batch):
            fit_list = []
            for body in pop:
                # init_spring(body, False)
                sim_lib.run(body, T, dt, osc)                    
                fit_list.append([body.fit,body])
            # fit_list = np.array(fit_list)
            
            pop = []
            for _ in range(pop_n):
                cross= selectionTour(fit_list,tourSize)
                mut = mutate(cross)
                pop.append(mut)
        print(f'{o*batch} interations completed')
        fit_list = np.array(fit_list)
        best_fit = fit_list[fit_list[:,0].argsort()][0][0]
        print(f'{best_fit} is the best fitness currently')
    best_body = fit_list[fit_list[:,0].argsort()][0][1]
    mass = best_body.mass
    spring = best_body.spring
    return mass, spring, best_body     
        
def savePickle(filename, list):
    with open(filename, "wb") as fp:
        pickle.dump(list, fp)

def openPickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

 
if __name__ == '__main__':
    
    l = 0.1
    m = 0.1
    n = 3


    T = 0.00
    dt = 0.00025
    # dt = 0.00025
    index = []
    lines = []
    vtx = []    
    spring_e = 0
    mass_e = 0
    osc = True

    mass_opt, spring_opt, body_opt = GA(T, dt, l, m, n)
    
    
    #3D visiualization
    springList = []
    for i in spring_opt:
        k = i.k
        b = i.b
        c = i.c
        springList.append([k,b,c])

    body = init_cube(l, m, n)
    mass = body.mass
    spring = body.spring
    for sp in springList:
        init_spring(body, False, sp[0], sp[2], sp[1])

    floor = box(pos = vector(0, -0.02, 0), color = color.white, length= 3, width = 3, height = 0.001)
    for sp in spring:
        # index.append((sp.m1_index,sp.m2_index))
        # lines.append([sp.m1.pos(), sp.m2.pos()])
        start = sp.m1.pos()
        vtr = sp.m2.pos()-start
        lines.append(cylinder(pos = vector(start[1], start[2], start[0]), axis = vector(vtr[1], vtr[2], vtr[0]), radius =0.01))
    for ms in mass:    
        # vertex.append(ms.pos())
        center = ms.pos()
        vtx.append(sphere(pos = vector(center[1], center[2], center[0]), radius = 0.02))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)
    x = []
    y = []
    yke = []
    ype = []
    y.append(spring_e+mass_e)
    yke.append(mass_e)
    ype.append(spring_e+mass_e)
    x.append(T)
    plot = ax.plot(x, y, animated=True, label = 'Total Energy [J]')[0]
    plotke = ax.plot(x, yke, animated=True, label = 'Kinetic Energy [J]')[0]
    plotpe = ax.plot(x, ype, animated=True, label = 'Potential Energy [J]')[0]
    ax.legend()
    plt.title(f'Bouncing cube: dt = {dt}, k = 7500, k_ground = 5000, damping = None')
    plt.xlabel('T')
    plt.ylabel('Energy [J]')
    plt.show(block=False)
    plt.pause(0.1)
    background = fig.canvas.copy_from_bbox(ax.bbox)
    ax.draw_artist(plot)
    fig.canvas.blit(fig.bbox)


    cycle_sec = []
    cycle_sec.append(0.00)
    t = []
    t.append(0.00)
    # for __ in range(80):
    while True:
        t1 = time.perf_counter()
        
        for _ in range(100):

                
            T += dt
            rate(200)
            if not osc:
                sim_lib.simulation(mass, spring, None, dt)
            else:
                sim_lib.simulation(mass, spring, T, dt)
            spring_e = 0
            mass_e = 0
            for i in range(len(spring)):
                start = spring[i].m1.pos()
                vtr = spring[i].m2.pos()-start
                lines[i].pos = vector(start[1], start[2], start[0])
                lines[i].axis = vector(vtr[1], vtr[2], vtr[0])
                spring_e += spring[i].pe()
                
            for i in range(len(mass)):    
                # vertex.append(ms.pos())
                center = mass[i].pos()
                vtx[i].pos = vector(center[1], center[2], center[0])
                mass_e += mass[i].ke()
                spring_e += mass[i].gpe()+ mass[i].flpe()
            
            
            
            y.append(spring_e+mass_e)
            yke.append(mass_e)
            ype.append(spring_e)
            x.append(x[-1]+dt)
            
            plot.set_data(x, y)
            plotke.set_data(x, yke)  
            plotpe.set_data(x, ype)
            
            ax.draw_artist(plot)
            ax.draw_artist(plotke)
            ax.draw_artist(plotpe)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            
        t2 = time.perf_counter()
        # cycle_sec.append(100/(t2-t1))
        # t.append(t[-1]+t2-t1)
        print(f'{100/(t2-t1)} cycles per seconds')

    # savePickle("cycles_per_second", cycle_sec)
    # savePickle("time_per_second", t)
    # print(index)
    # print(mass)
    # print(spring)
