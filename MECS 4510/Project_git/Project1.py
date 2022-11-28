import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from vpython import *
import concurrent.futures as cf
import time
import math
from random import randint
import pickle

class Mass(object):
    def __init__(self, m, p, v = np.array([0,0,0]), a = np.array([0,0,0]), f = np.array([0,0,0])):
        self.m = m
        self.p = p
        self.v = v
        self.a = a
        self.f = f
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
        
    def ke(self):
        k_e = 0.5*self.ma()*np.linalg.norm(self.v)**2
        return k_e
    
    def gpe(self):
        self.gp_e = self.ma()*9.81*self.pos()[2]
        return self.gp_e
    
    def flpe(self):
        return self.fl_pe

class Spring(object):
    def __init__(self, k, L0, m1_index, m2_index, m1 = 0, m2 = 0):
        self.k = k
        self.L0 = L0
        self.m1 = m1 
        self.m2 = m2 
        self.m1_index = m1_index
        self.m2_index = m2_index
        # self.c = randint(0,314)
        self.c = 0
        
    def spring_const(self):
        return self.k
    def length(self):
        return np.linalg.norm(self.m2.pos()-self.m1.pos())
    
    def generate_L0(self):
        self.L0 = np.linalg.norm(self.m2.pos()-self.m1.pos())
            
    def get_L0(self, value = None):
        if value:
            a = self.L0
            b = 0.005
            w = 100
            
            # c = 0
            return a + b*math.sin(w*value + self.c)
        else:
            return self.L0
            
    def len_v(self):
        return self.m2.pos()-self.m1.pos()
    def pe(self):
        pe = 0.5*self.spring_const()*np.linalg.norm(self.length()-self.L0)**2
        return pe

def cube(length, mass_corner):
    p = np.array([[0, 0, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 1, 0],
         [0, 0, 1],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 1],])
    
    r_m = np.array([[math.cos(0*(math.pi/180)), 0, math.sin(0*(math.pi/180))],
                    [0, 1, 0],
                    [-math.sin(0*(math.pi/180)), 0, math.cos(0*(math.pi/180))]])
    p_list = []
    for i in range(len(p)):
        p_list.append(np.matmul(r_m, np.transpose(p[i])))
    p = np.array(p_list)
    st = 2
    start_pos = np.array(
        [[0, 0, st],
         [0, 0, st],
         [0, 0, st],
         [0, 0, st],
         [0, 0, st],
         [0, 0, st],
         [0, 0, st],
         [0, 0, st],])
    p = p+start_pos
    p = p*length
    mass = []
    spring = []
    k = 7500
    # k = 7500
    for i in range(8):
        mass.append(Mass(mass_corner, p[i]))
    for j in range(1,8):
        for t in range(j,8):
            spring.append(Spring(k,0, j-1, t, mass[j-1],mass[t]))
    return mass, spring

def interaction(spring_list, mass_list, T):
    kc = 5000
    # kc = 5000
    
    for spring in spring_list:
        extension_f = spring.spring_const()*(spring.length()-spring.get_L0(T))
        dir = spring.len_v()/np.linalg.norm(spring.len_v())
        extension_f = extension_f*dir
        spring.m1.set_force(spring.m1.force()+extension_f)
        spring.m2.set_force(spring.m2.force()-extension_f)

    for mass in mass_list:
        if mass.pos()[2]<=0:
            mass.set_force(mass.force()+np.array([0,0,-kc*mass.pos()[2]]))
            # print(mass.pos()[2])
            mass.fl_pe = 0.5*kc*(mass.pos()[2])**2
        else:
            mass.set_force(mass.force()+mass.ma()*np.array([0,0,-9.81]))
            
def intergration(mass_list, dt):
    damp = 0.999
    # damp = 1
    for mass in mass_list:
        mass.set_acc(mass.force()/mass.ma())
        mass.set_vel((mass.vel()+mass.acc()*dt)*damp)
        mass.set_pos(mass.pos()+mass.vel()*dt)
        mass.set_force(0)

def simulation(mass, spring, T, dt):
   
    interaction(spring, mass, T)
    intergration(mass, dt)


def savePickle(filename, list):
    with open(filename, "wb") as fp:
        pickle.dump(list, fp)

def openPickle(filename):
    
    with open(filename, 'rb') as fp:
        return pickle.load(fp)



# class win(mglw.WindowConfig):
#     gl_version = (3, 3)
#     title = "ModernGL Example"
#     window_size = (1280, 720)
#     aspect_ratio = 16 / 9
#     resizable = True

#     resource_dir = os.path.normpath(os.path.join(__file__, '../../data'))

# ctx = mgl.create_standalone_context()

# prog = ctx.program(
#     vertex_shader='''
#         #version 330
#         in vec2 in_vert;
#         in vec3 in_color;
#         out vec3 v_color;
#         void main() {
#             v_color = in_color;
#             gl_Position = vec4(in_vert, 0.0, 1.0);
#         }
#     ''',
#     fragment_shader='''
#         #version 330
#         in vec3 v_color;
#         out vec3 f_color;
#         void main() {
#             f_color = v_color;
#         }
#     ''',
# )


 
mass, spring = cube(0.1, 0.1)
for i in range(len(spring)):
    spring[i].generate_L0()


floor = box(pos = vector(0, -0.02, 0), color = color.white, length= 1, width = 1, height = 0.001)
T = 0.00
dt = 0.00025
# dt = 0.00025
index = []
lines = []
vtx = []    
spring_e = 0
mass_e = 0
osc = False

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
            simulation(mass, spring, None, dt)
        else:
            simulation(mass, spring, T, dt)
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
