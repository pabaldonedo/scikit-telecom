import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class radiation_pattern(object):
    def __init__(self):
        self.data = np.zeros((1,1,1))

    def load_file(self, filename, comments_char='#', delimiter_char=None, etar=1):
        '''Loads data from txt file with the following columns: theta angle (degrees 0-180), phi angle 
           (degrees 0-360), amplitude of theta component of electric field (V/m), phase of theta component
           of electric field (degrees), amplitude of phi component of electric field (V/m),
           phase of phi component of electric field (degrees) 
        '''
        self.data = np.loadtxt(filename, comments=comments_char, delimiter=delimiter_char)
        self.etar = etar

    def load_data(self, data, etar=1):
        '''Loads data from numpy 2D-array with the following columns: theta angle (degrees 0-180), phi angle 
           (degrees 0-360), amplitude of theta component of electric field (V/m), phase of theta component
           of electric field (degrees), amplitude of phi component of electric field (V/m),
           phase of phi component of electric field (degrees) 
        '''
        self.data = data
        self.etar = etar

    def set_etar(self, etar):
        self.etar = etar

    def generate_mesh(self):
        '''Generates 3 matrix containing phi, theta coordinates and amplitude values
        '''
        phi, theta, e = self.data_to_spherical(self.data) 
        phigrid, thetagrid = np.meshgrid(np.unique(phi), np.unique(theta))
        rhogrid = np.zeros(phigrid.shape)
        for i in xrange(phigrid.shape[1]):
            rhogrid[:,i] = 20*np.log10(e[phi == phigrid[0, i]])
        offset = -np.min(rhogrid)
        return phigrid, thetagrid, rhogrid, offset

    def plot_uv(self):
        '''Plots the normalized radiation pattern in u-v coordinates
        '''
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

    def directivity(self):
        '''Computes the directivity of the antenna by integrating using the trapzedoial method
        '''
        eta = 120*np.pi*self.etar
        rad_power = self.radiated_power(eta)
        phi, theta, e = self.data_to_spherical(self.data) #TEMPORAL; must be changed!

        directivity = 4*np.pi*np.max(e**2)/(rad_power*eta*2)
        return 10*np.log10(directivity)

    def radiated_power(self, eta):
        '''Computes the radiated power by the antenna
        '''
        #Integral along theta
        phi, theta, e = self.data_to_spherical(self.data)
        phiangles = np.unique(phi)
        thetaangles = np.unique(theta)
        firstint = np.zeros(phiangles.shape)
        for i in xrange(phiangles.shape[0]):
            integrand = e[phi==phiangles[i]]**2*np.abs(np.sin(theta[phi==phiangles[i]]))
            firstint[i] = np.trapz(integrand, theta[phi==phiangles[i]])
        rad_power = 1.0/(2*eta)*np.trapz(firstint, phiangles)
        return rad_power

    def solid_angle(self):
        '''Computes the solid angle of the radiation pattern
        '''
        return 4.0*np.pi/10**(self.directivity()/10.0)

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

    def axial_ratio(self, ph=0, th=0):
        '''Computes axial ratio at phi angle = ph and theta angle = th
        '''
        phi, theta, _ = self.data_to_spherical(self.data)
        #complex values of the electrical field in both theta and phi components
        eth = self.data[:,2]*np.exp(1j*self.data[:,3]*np.pi/180.0)
        eph = self.data[:,4]*np.exp(1j*self.data[:,5]*np.pi/180.0)
        print eth
        #Decomposition in two circular waves
        erhc = 1.0/np.sqrt(2) * (eth+1j*eph)
        elhc = 1.0/np.sqrt(2) * (eth-1j*eph)
        #index of value at phi= ph theta = th
        index = (phi==ph) & (theta==th)
        #axial ratio in ph, th
        r = np.divide(float(np.abs(erhc[index])+np.abs(elhc[index])), (np.abs(erhc[index])-np.abs(elhc[index])))
        return 20*np.log10(r)

    def sph2cart(self, phi, theta, rho):
        '''Transforms spherical coordinates to cartesians
        '''
        x = rho*np.sin(theta)*np.cos(phi)   
        y = rho*np.sin(theta)*np.sin(phi)
        z = rho*np.cos(theta)
        return x, y, z
