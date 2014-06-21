import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class radiation_pattern(object):
    def __init__(self):
        self.data = np.zeros((1,1,1))

    def load_file(filename, comments_char='#', delimiter_char=','):
        '''Loads data from txt file with the following columns: theta angle (degrees), phi angle 
           (degrees), amplitude of theta component of electric field (V/m), phase of theta component
           of electric field (degrees), amplitude of phi component of electric field (V/m),
           phase of phi component of electric field (degrees) 
        '''
        self.data = np.loadtxt(filename, comments=comments_char, delimiter=delimieter_char)

    def load_data(data):
        '''Loads data from numpy 2D-array with the following columns: theta angle (degrees), phi angle 
           (degrees), amplitude of theta component of electric field (V/m), phase of theta component
           of electric field (degrees), amplitude of phi component of electric field (V/m),
           phase of phi component of electric field (degrees) 
        '''
        self.data = data

    def plot():
        '''Plots the normalized radiation pattern
        '''
        phi, theta, e = self.data_to_spherical(self.data) 
        phigrid, thetagrid = np.meshgrid(np.unique(phi), np.unique(theta))
        rhogrid = np.zeros(phigrid.shape)
        for i in xrange(phigrid.shape[1]):
            rhogrid[:,i] = 20*np.log10(e[phi == phigrid[0, i]])
        offset = -np.min(rhogrid)
        x, y, z = sph2cart(phigrid, thetagrid, rhogrid+offset)
        z -= np.max(z)
        fig = plt.figure()
        controller = fig.gca(projection='3d')
        surface = controller.plot_surface(y, x, z, rstride=1, cstride=1, cmap=cm.jet)
        fig.colorbar(surface, ticks = np.arange(0, np.min(z), -10))
        plt.show()  

    def data_to_spherical(data):
        '''Computes the amplitude of the radiation pattern and transforms the angles from degrees
           to radians
        '''
        theta = data[:,0]*np.pi/180
        phi = data[:,1]*np.pi/180
        e_theta = data[:,2]
        e_phi = data[:,4]
        e = np.sqrt(e_theta**2+e_phi**2)
        return phi, theta, e

    def sph2cart(phi, theta, rho):
        '''Transforms spherical coordinates to cartesians
        '''
        x = rho*np.sin(theta)*np.cos(phi)   
        y = rho*np.sin(theta)*np.sin(phi)
        z = rho*np.cos(theta)
        return x, y, z
