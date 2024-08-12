import numpy as np
from matplotlib import pyplot as plt
import sys

N=1000 #Number of particles
nu=np.pi/4 #Parameter used in the scheme
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
    return (a,tetha)

def invert_coords(coords):
    """Gets the euclidian coordinates for an array of points belonging in a same C(a)"""
    a=coords[0][0]
    tethas=coords.T[1].T
    x=np.abs(a)*np.sin(tethas)
    y=a*(1-np.cos(tethas))
    return(x,y)

def concatenate(array1,array2):
    """This function generalizes the numpy.concatenate method even when one of the array is empty"""
    if len(array1)==0:
        return array2
    if len(array2)==0:
        return array1
    return np.concatenate((array1,array2))

# We implement a sorting algorithm, which enables to simplify the implementation of the dynamics by sorting the particles according to their angle
def merge_sort(array):
    """
    Sorts the lines of a 2D array according to the first components using the merge sort algorithm
    
    Parameters:
        array (np.array): a 2D array which represents the particles angle and their weight for a specific circle C(a)
        
    Returns:
        An updated array: every angle is set between 0 and 2pi and the array is sorted"""
    
    def merge(array,array1,array2):
        """Merges two sorted arrays"""
        if len(array1)==0:
            return concatenate(array,array2)
        if len(array2)==0:
            return concatenate(array,array1)
        if array1[0][1]<array2[0][1]:
            return merge(concatenate(array,array1[:1][:][:]),array1[1:][:][:],array2)
        return merge(concatenate(array,np.array(array2[:1][:])),array1,array2[1:][:])
    
    def sort(array):
        """Splits the array in half to implement to merge sort algorithm"""
        if len(array)==0:
            return array
        if len(array)==1:
            array[0][1]=array[0][1]-2*np.pi*(np.floor(array[0][1]/(2*np.pi)))
            return array
        return merge(np.array([]),sort(array[len(array)//2:]),sort(array[:len(array)//2]))
    return sort(array)

# We determine the initial particle distribution using a gaussian distribution centered around pi
positions_x=2.*np.random.normal(loc=0,scale=1,size=(N))
positions_y=2.*np.random.normal(loc=0,scale=1,size=(N))
plt.scatter(positions_x,positions_y)
plt.show()

circular_coordinates=[change_coords(positions_x[i],positions_y[i]) for i in range(N)]

#Then we define the particles, with an initial weight of 1. This changes whenever a particle is split in half by the control
particles=[np.array([[a,tetha,1.]]) for a,tetha in circular_coordinates]

#In the 2D case, initially each particle is associated to its own circle C(a). Yet, when splitted, this circle will be populated
# with new particles. Therefore the "particles" list below contains the different orbits. Initially each orbit contains 1 particle.

#Here this function looks for the limits of the A set, and looks for tetha_nu. Here we take advantage of the particles being sorted
def look_for_tetha(particles):
    """
    Look for the bounds of the particles set belonging to A(a) -- ie the indexs of the particles whose angle is higher than pi-nu and lesser than pi+mu

    Parameters:
        particles(np.array): the 2D array containing the angles and the weights of the particles at a given iteration
    
    Returns:
        A list of three elements:
            - the highest index of the particles whose angle is lesser than pi-nu, None if all have an higher angle
            - the lowest index of the particles whose angle is higher than pi+nu, None if all have an lower angle
            - a tuple (index,ratio) where index is the index where half of the mass of set A is reached, and ratio is c1
    """
    l=len(particles)
    start,end=0,l-1
    if particles[end][1]<np.pi-nu:
        #All the particles angles are lesser than pi-nu, the dynamics are trivial
        return [end,None,None]
    if particles[start][1]>np.pi+nu:
        #All the particles angles are higher than pi+nu, the dynamics are also trivial
        return [None,start,None]
    
    #We do a first dichotomy to find the inferior bound of set A
    while start<end-1:
        #we look for the inferior bound of the set A
        m=(start+end)//2
        if particles[m][1]<np.pi-nu:
            start=m
        else:
            end=m
            
    left_i=start
    start,end=0,l-1
    #We do another dichotomy to find the superior bound of set A
    while start<end-1:
        #we look for the superior bound of the set A
        m=(start+end)//2
        if particles[m][1]>np.pi+nu:
            end=m
        else:
            start=m
    right_i=end

    if right_i-left_i<=1:
        if  particles[left_i][1]<np.pi-nu and particles[right_i][1]>np.pi+nu:
        #In this case, it means the set A is empty
            return [left_i,right_i,None]
        else:
            if particles[left_i][1]>np.pi-nu:
                left_i-=1
            if particles[right_i][1]>np.pi-nu:
                right_i+=1
    
    # We compute the total weight of the set A
    mass=np.sum(particles[left_i+1:right_i],axis=0)[2]
    it=left_i+1
    m=particles[left_i+1][2]
    while m< mass/2:
        it+=1
        m+=particles[it][2]
    return [left_i,right_i,(it,(m-mass/2)/particles[it][2])]

def stabilization(particles_1):
    """
    Generates the next iteration from the previous one

    Parameter:
        particles_1 (np.array): a 2D array containing the angles and the weights of the particles
    """
    new_particles=[]
    for array in particles_1:
        #For every orbit, we look for the A set and the angle where half of its mass is reached
        a=array[0][0]
        l=len(array)
        particles=np.copy(array)
        bounds=look_for_tetha(particles)
        x,y=invert_coords(array)
        factor=np.exp(-np.reciprocal(x**2+y**2))/np.abs(a)

        if bounds[2] is not None:
            #If A is non-empty
            particles.T[1][np.arange(l)<bounds[2][0]]-=factor[np.arange(l)<bounds[2][0]]*dt
            particles.T[1][np.arange(l)>bounds[2][0]]+=factor[np.arange(l)>bounds[2][0]]*dt
            if particles[bounds[2][0]][2]>0.01:
                # We only care to split a particle in half if its weight is higher than 0.01
                if bounds[2][1]==0 or bounds[2][1]==1:
                    # In some cases, if c1=0 or c2=0, we don't need to split the particle
                    particles[bounds[2][0]][1]-=factor[bounds[2][0]]*dt*(2*bounds[2][1]-1)
                else:
                    # else, a proportion of c1 goes in a way and c2 goes the other way
                    particles[bounds[2][0]][1],particles[bounds[2][0]][2]=particles[bounds[2][0]][1]-factor[bounds[2][0]]*dt,bounds[2][1]*particles[bounds[2][0]][2]
                    particles=concatenate(particles,np.array([[a,particles[bounds[2][0]][1]+factor[bounds[2][0]]*dt,(1-bounds[2][1])*particles[bounds[2][0]][2]]]))
            else:
                # If the particle's weight is too small, we send it in a given direction
                particles.T[1][bounds[2][0]]-=factor[bounds[2][0]]*dt
        elif bounds[0] is None:
            # If A is empty and every angle is higher than pi+nu, all particles turn clockwise
            particles.T[1]+=factor*dt
        elif bounds[1] is None:
            # If A is empty and every angle is lesser than pi-nu, every particle goes in the other direction
            particles.T[1]-=factor*dt
        else:
            # Last case, where some angles are lesser than pi-nu and the other are higher than pi+nu
            particles.T[1][np.arange(l)<=bounds[0]]-=factor[np.arange(l)<=bounds[0]]*dt
            particles.T[1][np.arange(l)>=bounds[1]]+=factor[np.arange(l)>=bounds[1]]*dt
        # We make sure the new iteration is sorted
        new_particles.append(merge_sort(particles))
    return new_particles

part_evolution=[particles]
for i in range(2000):
    # We compute a certain amount of iterations
    part_evolution.append(stabilization(part_evolution[-1]))


""""""
#The following code just enables to display the evolution of the particles

from celluloid import Camera
from numpy import genfromtxt
camera = Camera(plt.figure())



for i in range(2000):
    x=np.array([])
    y=np.array([])
    weights=np.array([])
    for particle_orbite in part_evolution[i]:
        x_coords,y_coords=invert_coords(particle_orbite)
        x=concatenate(x,x_coords)
        y=concatenate(y,y_coords)
        weights=concatenate(weights,particle_orbite.T[2].T+0.001)
    plt.scatter(x,y,c=weights,cmap="Blues",norm="log",vmin=0.001,vmax=1)
    camera.snap()
anim = camera.animate(blit=True)
anim.save('2D_gaussian_distribution_with_splitting1.gif',fps=500)

print(len(x_coords))


"""
camera = Camera(plt.figure())


for i in range(5000):
    mat=np.zeros((50,50))
    for j in range(len(part_evolution[i])):
        mat[int(25*(1-np.cos(part_evolution[i][j][0]))),int(25*(1-np.cos(part_evolution[i][j][0])))]+=part_evolution[i][j][1]
    plt.imshow(mat)
    camera.snap()
anim = camera.animate(blit=True)
anim.save('gaussian_distribution_with_splitting2.gif', fps=100)
"""