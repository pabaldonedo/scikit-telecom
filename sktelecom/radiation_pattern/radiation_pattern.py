import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class radiation_pattern(object):
    def __init__(self):
        self.data = np.zeros((1,1,1))

    def load_file(self, filename, comments_char='#', delimiter_char=None):
        '''Loads data from txt file with the following columns: theta angle (degrees), phi angle 
           (degrees), amplitude of theta component of electric field (V/m), phase of theta component
           of electric field (degrees), amplitude of phi component of electric field (V/m),
           phase of phi component of electric field (degrees) 
        '''
        self.data = np.loadtxt(filename, comments=comments_char, delimiter=delimiter_char)

    def load_data(self, data):
        '''Loads data from numpy 2D-array with the following columns: theta angle (degrees), phi angle 
           (degrees), amplitude of theta component of electric field (V/m), phase of theta component
           of electric field (degrees), amplitude of phi component of electric field (V/m),
           phase of phi component of electric field (degrees) 
        '''
        self.data = data

    def generate_mesh(self):

        phi, theta, e = self.data_to_spherical(self.data) 
        phigrid, thetagrid = np.meshgrid(np.unique(phi), np.unique(theta))
        rhogrid = np.zeros(phigrid.shape)
        for i in xrange(phigrid.shape[1]):
            rhogrid[:,i] = 20*np.log10(e[phi == phigrid[0, i]])
        offset = -np.min(rhogrid)
        return phigrid, thetagrid, rhogrid, offset

    def plot_uv(self):
        phigrid, thetagrid, rhogrid, offset = self.generate_mesh()
        u = np.sin(thetagrid)*np.cos(phigrid)
        v = np.sin(thetagrid)*np.sin(phigrid)
        z = rhogrid + offset 
        z -= np.max(z)
        fig = plt.figure()
        controller = fig.gca(projection='3d')
        surface = controller.plot_surface(u, v, z, rstride=1, cstride=1, cmap=cm.jet)
        plt.xlabel('u')
        plt.ylabel('v')
        fig.colorbar(surface, ticks=np.arange(0, np.min(z), -10), label='Directivity [dB]')
        plt.show()


    def plot3d(self):
        '''Plots the normalized radiation pattern
        '''
        phigrid, thetagrid, rhogrid, offset = self.generate_mesh()
        x, y, z = self.sph2cart(phigrid, thetagrid, rhogrid+offset)
        z -= np.max(z)
        fig = plt.figure()
        controller = fig.gca(projection='3d')
        surface = controller.plot_surface(y, x, z, rstride=1, cstride=1, cmap=cm.jet)
        plt.xlabel('x')
        plt.ylabel('z')
        fig.colorbar(surface, ticks=np.arange(0, np.min(z), -10), label='Directivity [dB]')
        plt.show()  

    def data_to_spherical(self, data):
        '''Computes the amplitude of the radiation pattern and transforms the angles from degrees
           to radians
        '''
        theta = data[:,0]*np.pi/180
        phi = data[:,1]*np.pi/180
        e_theta = data[:,2]
        e_phi = data[:,4]
        e = np.sqrt(e_theta**2+e_phi**2)
        return phi, theta, e

    def sph2cart(self, phi, theta, rho):
        '''Transforms spherical coordinates to cartesians
        '''
        x = rho*np.sin(theta)*np.cos(phi)   
        y = rho*np.sin(theta)*np.sin(phi)
        z = rho*np.cos(theta)
        return x, y, z
