import numpy as np
from matplotlib import pyplot as plt
import sys
import os

N=1000 #Number of particles
nu=np.pi/4 #Parameter used in the scheme
dt=0.01 #time step

sys.setrecursionlimit(N*N) #This is needed for the merge sort algorithm which uses recursion

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
        array (np.array): a 2D array which represents the particles angle and their weight
        
    Returns:
        An updated array: every angle is set between 0 and 2pi and the array is sorted"""
    
    def merge(array,array1,array2):
        """Merges two sorted arrays"""
        if len(array1)==0:
            return concatenate(array,array2)
        if len(array2)==0:
            return concatenate(array,array1)
        if array1[0]<array2[0]:
            return merge(concatenate(array,array1[:1]),array1[1:],array2)
        return merge(concatenate(array,np.array(array2[:1])),array1,array2[1:])
    
    def sort(array):
        """Splits the array in half to implement the merge sort algorithm"""
        if len(array)==0:
            return array
        if len(array)==1:
            array[0]=array[0]-2*np.pi*(np.floor(array[0]/(2*np.pi)))
            return array
        return merge(np.array([]),sort(array[len(array)//2:]),sort(array[:len(array)//2]))
    return sort(array)

# We determine the initial particle distribution using a gaussian distribution centered around pi (any other distribution can be chosen)
particles=2*np.pi*np.random.exponential(scale=0.1,size=N)
particles-=2*np.pi*np.floor(particles/(2*np.pi))

#Here this function looks for the limits of the A set, and looks for tetha_nu. Here we take advantage of the particles being sorted
def look_for_tetha(particles):
    """
    Look for the bounds of the particles set belonging to A -- ie the indexs of the particles whose angle is higher than pi-nu and lesser than pi+mu

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
    if particles[end]<np.pi-nu:
        #All the particles angles are lesser than pi-nu, the dynamics are trivial
        return [end,None,None]
    if particles[start]>np.pi+nu:
        #All the particles angles are higher than pi+nu, the dynamics are also trivial
        return [None,start,None]
    
    #We do a first dichotomy to find the inferior bound of set A
    while start<end-1:
        #we look for the inferior bound of the set A
        m=(start+end)//2
        if particles[m]<np.pi-nu:
            start=m
        else:
            end=m
            
    left_i=start
    start,end=0,l-1
    #We do another dichotomy to find the superior bound of set A
    while start<end-1:
        #we look for the superior bound of the set A
        m=(start+end)//2
        if particles[m]>np.pi+nu:
            end=m
        else:
            start=m
    right_i=end

    if right_i-left_i<=1:
        if  particles[left_i]<np.pi-nu and particles[right_i]>np.pi+nu:
        #In this case, it means the set A is empty
            return [left_i,right_i,None]
        else:
            if particles[left_i]>np.pi-nu:
                left_i-=1
            if particles[right_i]>np.pi-nu:
                right_i+=1
    return [left_i,right_i,(right_i+left_i)//2]

def stabilization(particles_1):
    """
    Generates the next iteration from the previous one

    Parameter:
        particles_1 (np.array): a 2D array containing the angles and the weights of the particles
    """
    l=len(particles_1)
    particles=np.copy(particles_1)
    bounds=look_for_tetha(particles)

    if bounds[2] is not None:
        #If A is non-empty
        particles[np.arange(l)<=bounds[2]]-=dt
        particles[np.arange(l)>bounds[2]]+=dt
    elif bounds[0] is None:
        # If A is empty and every angle is higher than pi+nu, all particles turn clockwise
        particles+=dt
    elif bounds[1] is None:
        # If A is empty and every angle is lesser than pi-nu, every particle goes in the other direction
        particles-=dt
    else:
        # Last case, where some angles are lesser than pi-nu and the other are higher than pi+nu
        particles[np.arange(l)<=bounds[0]]-=dt
        particles[np.arange(l)>=bounds[1]]+=dt
    # We make sure the new iteration is sorted
    return merge_sort(particles)

part_evolution=[particles]
for i in range(500):
    # We compute a certain amount of iterations
    part_evolution.append(stabilization(part_evolution[-1]))


wasserstein1=[np.sum(np.abs(np.minimum(particles,2*np.pi-particles)))/N for particles in part_evolution]
wasserstein2=[np.sqrt(np.sum(np.power(np.minimum(particles,2*np.pi-particles),2))/N) for particles in part_evolution]


plt.plot(wasserstein1,color="blue")
plt.plot(wasserstein2,color="red")
plt.legend(["Wasserstein distance 1","Wasserstein distance 2"])
plt.savefig(
            os.path.join("./", "wasserstein_distances_exponential distribution no splitting.png"),
            dpi=300,
            bbox_inches="tight",
        )
plt.show()

""""""
#The following code just enables to display the evolution of the particles

from celluloid import Camera
from numpy import genfromtxt
camera = Camera(plt.figure())

for i in range(500):
    plt.scatter(-np.cos(part_evolution[i]),-np.sin(part_evolution[i]),color="blue")
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    camera.snap()
anim = camera.animate(blit=True)
anim.save('exponential_distribution_without_splitting1.gif', fps=100)


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