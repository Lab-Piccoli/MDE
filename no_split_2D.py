import numpy as np
from matplotlib import pyplot as plt
import sys

N=1000 #Number of particles
dt=0.01 #time step

sys.setrecursionlimit(N*N)

def change_coords(x,y):
    """
    Converts the euclidian coordinates into C(a) coordinates (a,angle)

    Parameters: 
        x (float): abcissa coordinate
        y (float): ordinate coordinate
    
    Returns:
        a (float): a such as the point (x,y) is contained by C(a)
        tetha (float): the angle of the point in C(a)
    """
    a=(x**2+y**2)/(2*y)
    tetha=np.angle(1-y/a+x/np.abs(a)*1.j)
    tetha-=np.floor(tetha/(2*np.pi))*2*np.pi
    return [a,tetha]

def invert_coords(coords):
    """Gets the euclidian coordinates for an array of points belonging in a same C(a)"""
    a=coords[0]
    tethas=coords[1]
    x=np.abs(a)*np.sin(tethas)
    y=a*(1-np.cos(tethas))
    return(x,y)

# We implement a sorting algorithm, which enables to simplify the implementation of the dynamics by sorting the particles according to their angle


# We determine the initial particle distribution using a gaussian distribution centered around pi
positions_x=2.*np.random.normal(loc=0,scale=1,size=(N))
positions_y=2.*np.random.normal(loc=0,scale=1,size=(N))
plt.scatter(positions_x,positions_y)
plt.show()

circular_coordinates=[change_coords(positions_x[i],positions_y[i]) for i in range(N)]

#Then we define the particles, with an initial weight of 1. This changes whenever a particle is split in half by the control
particles=[[a,tetha] for a,tetha in circular_coordinates]

#Here this function looks for the limits of the A set, and looks for tetha_nu. Here we take advantage of the particles being sorted

def stabilization(particles_1):
    """
    Generates the next iteration from the previous one

    Parameter:
        particles_1 (np.array): a 2D array containing the angles and the weights of the particles
    """
    new_particles=[]
    for particle_ in particles_1:
        particle=np.copy(particle_)
        a=particle[0]
        x,y=invert_coords(particle)
        factor=np.exp(-np.reciprocal(x**2+y**2))/np.abs(a)
        if particle[1]<np.pi:
            particle[1]-=factor*dt
        else:
            particle[1]+=factor*dt
        particle[1]-=2*np.pi*np.floor(particle[1]/(2*np.pi))
        new_particles.append(particle)
    return new_particles

part_evolution=[particles]
for i in range(2000):
    # We compute a certain amount of iterations
    part_evolution.append(stabilization(part_evolution[-1]))


""""""
#The following code just enbales to display the evolution of the particles

from celluloid import Camera
from numpy import genfromtxt
camera = Camera(plt.figure())



for i in range(2000):
    x=[]
    y=[]
    weights=np.array([])
    for particle_orbite in part_evolution[i]:
        x_coords,y_coords=invert_coords(particle_orbite)
        x.append(x_coords)
        y.append(y_coords)
    plt.scatter(x,y,color="blue")
    camera.snap()
anim = camera.animate(blit=True)
anim.save('2D_gaussian_distribution_without_splitting1.gif',fps=500)


"""
camera = Camera(plt.figure())


for i in range(5000):
    mat=np.zeros((50,50))
    for j in range(len(part_evolution[i])):
        mat[int(25*(1-np.cos(part_evolution[i][j][0]))),int(25*(1-np.cos(part_evolution[i][j][0])))]+=part_evolution[i][j][1]
    plt.imshow(mat)
    camera.snap()
anim = camera.animate(blit=True)
anim.save('gaussian_distribution_without_splitting2.gif', fps=100)
"""