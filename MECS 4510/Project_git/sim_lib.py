import numpy as np
import multiprocessing as mp
import os

def run(body, T, dt, osc):
    body.get_cm0()
    for _ in range(10):
        T += dt
        # if not osc:
        #     simulation(body.mass, body.spring, None, dt)
        # else:
        p, p_iter = simulation(body.mass, body.spring, T, dt)
        [x.start() for x in p]
        [y.start() for y in p_iter]
    body.get_cm()
    body.fitness()
    
def simulation(mass_list, spring_list, T, dt):
    process = []
    p_inter = []
    for spring in spring_list:
        p = mp.Process(target = interaction_sp, args=(spring, T))
        process.append(p)
    for mass in mass_list:
        p = mp.Process(target = interaction_m, args=(mass,))
        process.append(p)
    for mass in mass_list:
        p = mp.Process(target = intergration, args=(mass,dt))
        p_inter.append(p)
    return process, p_inter

def intergration(mass, dt):
    damp = 0.999
    # damp = 1
    # for mass in mass_list:
    mass.set_acc(mass.force()/mass.ma())
    mass.set_vel((mass.vel()+mass.acc()*dt)*damp)
    mass.set_pos(mass.pos()+mass.vel()*dt)
    mass.set_force([0,0,0])
        
def interaction_sp(spring, T):
    # for spring in spring_list:
    extension_f = spring.spring_const()*(spring.length()-spring.get_L0(T))
    dir = spring.len_v()/np.linalg.norm(spring.len_v())
    extension_f = extension_f*dir
    spring.m1.set_force(spring.m1.force()+extension_f)
    spring.m2.set_force(spring.m2.force()-extension_f)

def interaction_m(mass):
    kc = 5000
    mu = 0.25
    # for mass in mass_list:
    if mass.pos()[2]<=0:
        F_p = mass.force()
        if mass.pos()[2]<0:
            F_p = F_p+np.array([0,0,-kc*mass.pos()[2]])
        F_n = mass.ma()*np.array([0,0,9.81])
        if np.linalg.norm(F_p)<np.linalg.norm(F_n)*mu:
            F_p[0] = 0
            F_p[1] = 0
        if np.linalg.norm(F_p)>=np.linalg.norm(F_n)*mu:
            F_p[0] -= np.linalg.norm(F_n)*mu
            F_p[1] -= np.linalg.norm(F_n)*mu
        mass.set_force(F_p)
        # print(mass.pos()[2])
        mass.fl_pe = 0.5*kc*(mass.pos()[2])**2
            
    else:
        mass.set_force(mass.force()+mass.ma()*np.array([0,0,-9.81]))
        
# if __name__ == '__main__':
    