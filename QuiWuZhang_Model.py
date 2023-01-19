# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:42:46 2022
@authors: Josep Mas Garcia, Alejandro Molina Sánchez
This is a program to generate the calculations and plots for the QWZ model
"""

from numpy import sin, cos, pi, linalg, zeros, log, array, arange, sqrt, shape, meshgrid, exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter, MultipleLocator

class Ribbon(object):
      """
      Class to generate the hamiltonian of the QWZ model for the ribbon and to derive several
      results.
      """
      def __init__(self,n_cell):
          """
          Function that defines the basic parameters for the model. We manually introduce n_cell
          """
          self.n_sigma = 2           # Dimension sigma matrix
          self.n_cell = n_cell        # Number of unit cells

          self.sigma_x = array([[0.0, 1.0], [1.0,0.0]])     #Pauli matrices
          self.sigma_y = array([[0.0, -1.0j], [1.0j,0.0]])
          self.sigma_z = array([[1.0, 0.0], [0.0,-1.0]])


      def hamiltonian_k(self,kpoint,u_parameter):
          """
          Function that defines the ribbon's hamiltonian. We manually introduce
          kpoint and u_parameter
          """
          self.u     = u_parameter   # u parameter
          self.hamiltonian = zeros([self.n_cell*self.n_sigma,self.n_cell*self.n_sigma],dtype=complex)

          for i in range(self.n_cell):
              l=i+1
              i1 = int(i*self.n_sigma)
              i2 = int(i*self.n_sigma+self.n_sigma)
              
              self.hamiltonian[i1:i2,i1:i2]=(cos(kpoint)+self.u)*self.sigma_z+sin(kpoint)*self.sigma_y

              if i < self.n_cell-1:
                 i1  = int(i*self.n_sigma)
                 i2  = int(i*self.n_sigma+self.n_sigma)
                 ip1 = int(l*self.n_sigma)
                 ip2 = int(l*self.n_sigma+self.n_sigma)
                 
                 self.hamiltonian[ip1:ip2,i1:i2]=1.0*(self.sigma_z+1.0j*self.sigma_x)*0.5
                 self.hamiltonian[i1:i2,ip1:ip2]=1.0*(self.sigma_z-1.0j*self.sigma_x)*0.5

          return self.hamiltonian


      def plot_bands(self,n_kpoints,u_parameter):
          """
          Function that plots the band spectrum with color map in the first Brillouin zone.
          We manually introduce the number of kpoints for the representation and the u value
          """
          k_step = 2.0*pi/n_kpoints
          self.kpoints = arange(-pi,pi+k_step,k_step)       #First Brillouin Zone
          
          
          eigen=zeros([n_kpoints+1,self.n_cell*self.n_sigma])
          eigv=zeros([n_kpoints+1,self.n_cell*self.n_sigma,self.n_cell*self.n_sigma],dtype=(complex))
          
          for iky,k_y in enumerate(self.kpoints):
              hamiltonian=Ribbon(self.n_cell).hamiltonian_k(k_y,u_parameter)
              eig, wf = linalg.eigh(hamiltonian)
              eigen[iky,:] =eig
              eigv[iky,:,:]=wf
              
              w = zeros([n_kpoints+1,self.n_cell*self.n_sigma])   #color mapping for surface states
              
              for i in range(self.n_cell*self.n_sigma):
                  w[iky,i] = (abs(eigv[iky,0,i])**2+abs(eigv[iky,1,i])**2) - (abs(eigv[iky,-1,i])**2+abs(eigv[iky,-2,i])**2)
                  
          for i in range(self.n_cell*self.n_sigma): 
              plt.plot(self.kpoints,eigen[:,i],'r-')

          plt.xlabel(r'k$_{y}$',fontsize='x-large',fontstyle='italic',fontfamily='serif')

          plt.ylabel(r'Energia, E',fontsize='x-large',fontstyle='italic',fontfamily='serif')

          plt.show()
          
          for i in range(self.n_cell*self.n_sigma):
              plt.scatter(self.kpoints,eigen[:,i],s=2,c=w[:,i],cmap='viridis')
          
          plt.xlabel(r'k$_{y}$',fontsize='x-large',fontstyle='italic',fontfamily='serif')
          plt.ylabel(r'Energia, E',fontsize='x-large',fontstyle='italic',fontfamily='serif')
          plt.show()
          
          
      def location(self,kpoint,u_parameter):
          """
          Function that plots the localization probablity in the unit cells of real space of the zero energy states
          """
          hamiltonian=Ribbon(self.n_cell).hamiltonian_k(kpoint,u_parameter)
          self.eig, self.eigv = linalg.eigh(hamiltonian)
          
          for i in range(self.n_cell):
              fig1=plt.bar(i,abs(self.eigv[2*i,self.n_cell-1])**2+abs(self.eigv[2*i+1,self.n_cell-1])**2,width=1, color=['y'])
              fig2=plt.bar(i,abs(self.eigv[2*i,self.n_cell])**2+abs(self.eigv[2*i+1,self.n_cell])**2,width=1, color=['b'])
              fig3=plt.bar(i,abs(self.eigv[2*i,-1])**2+abs(self.eigv[2*i+1,-1])**2,width=1, color=['g'])
              
          plt.xlabel(r'n$_{site}$',fontsize='xx-large',fontstyle='italic',fontfamily='serif')
          plt.ylabel(r'Probabilitat',fontsize='xx-large',fontstyle='italic',fontfamily='serif')
          plt.legend([fig1,fig2,fig3],['banda N', 'banda N+1', 'banda 2N'])
          plt.show()   
          
          
      def diagram(self,u_step):
          
          u_values = arange(-3.0,3+u_step,u_step)
          
          chi1, chi2, chi11, chi22 = [], [], [], []
          chi3, chi4 = [], []

          ss_0 = zeros([len(u_values),self.n_sigma*self.n_cell])
          ss_pi = zeros([len(u_values),self.n_sigma*self.n_cell])

          for iu,u in enumerate(u_values):
              '''
              Bulk's energy spectrum in the Dirac Points
              '''
              chi1.append(sqrt(sin(0)**2+sin(0)**2+(u+cos(0)+cos(0))**2))
              chi2.append(-sqrt((sin(0))**2+(sin(0))**2+(u+cos(0)+cos(0))**2))
              chi11.append(sqrt(sin(0)**2+sin(0)**2+(u+cos(pi)+cos(pi))**2) )
              chi22.append( -sqrt((sin(0))**2+(sin(0))**2+(u+cos(pi)+cos(pi))**2) )
              chi3.append(sqrt(sin(0)**2+sin(pi)**2+(u+cos(pi)+cos(0))**2) )
              chi4.append( -sqrt((sin(0))**2+(sin(pi))**2+(u+cos(pi)+cos(0))**2) )
              
              '''
              Ribbon's energy spectrum in k_y=0
              '''
              hamiltoniancz=Ribbon(self.n_cell).hamiltonian_k(0,u) 
              eigcz, wfcz = linalg.eigh(hamiltoniancz)
              ss_0[iu,:] = eigcz
               
              '''
              Ribbon's energy spectrum in k_y=pi
              '''
              hamiltonianvz = Ribbon(self.n_cell).hamiltonian_k(pi,u) 
              eigvz, wfvz = linalg.eigh(hamiltonianvz)
              ss_pi[iu,:] = eigvz

            
          '''
          Bulk's bands
          '''
          plt.plot(u_values,chi1,'k-',lw=1)
          plt.plot(u_values,chi2,'k-',lw=1)
          plt.plot(u_values,chi3,'k-',lw=1)
          plt.plot(u_values,chi4,'k-',lw=1)
          plt.plot(u_values,chi11,'k-',lw=1)
          plt.plot(u_values,chi22,'k-',lw=1)

          plt.plot(u_values,ss_0[:,self.n_cell],'m-')
          plt.plot(u_values,ss_0[:,self.n_cell-1],'m-')

          plt.plot(u_values,ss_pi[:,self.n_cell],'g-')
          plt.plot(u_values,ss_pi[:,self.n_cell-1],'g-')

          plt.xlabel(r'Paràmetre, u',fontsize='x-large',fontstyle='italic',fontfamily='serif')
          plt.ylabel(r'Energia, E',fontsize='x-large',fontstyle='italic',fontfamily='serif')

          #plt.ylim(-2,2)
          plt.ylim(-3,3)
          plt.show()
          
class bulk(object):
    """
    Class to generate the band spectrum of the bulk of the QWZ model for the bulk and to derive several
    results.
    """
    def __init__(self,u_value):
        """
        Function to generate the bulk's u_value and Brillouin Zone
        """
        self.u_value=u_value
        
        
        
    def energy(self,n_kpoints):
        k_step = 2.0*pi/n_kpoints
        self.kpoints = arange(-pi,pi+k_step,k_step)
        X,Y=meshgrid(self.kpoints,self.kpoints)
        """
        Function that plots the two-band spectrum for the bulk
        """
        chi1 = sqrt(sin(X)**2+sin(Y)**2+(self.u_value+cos(X)+cos(Y))**2)
        chi2 = -sqrt((sin(X))**2+(sin(Y))**2+(self.u_value+cos(X)+cos(Y))**2)
        """
        General solution for any u_value for the two-band spectrum of the QWZ model
        """
        fig = plt.figure(figsize = (5,5))
        ax = Axes3D(fig)

        ax.plot_surface(X, Y, chi1)
        ax.plot_surface(X, Y, chi2)
        plt.xlabel(r'k$_{x}$',fontsize='xx-large',fontstyle='italic',fontfamily='serif')
        plt.ylabel(r'k$_{y}$',fontsize='xx-large',fontstyle='italic',fontfamily='serif',rotation=0)

        ax.xaxis.set_major_formatter(FuncFormatter(
           lambda val,pos: '$\pi$'.format(val/pi) if val !=0 else '0'
        ))
        ax.xaxis.set_major_locator(MultipleLocator(base=pi))
        ax.yaxis.set_major_formatter(FuncFormatter(
           lambda val,pos: '$\pi$'.format(val/pi) if val !=0 else '0'
        ))
        ax.yaxis.set_major_locator(MultipleLocator(base=pi))

        plt.figtext(0.10, 0.9, 'u=2',fontsize='30',fontstyle='italic',fontfamily='serif') #hay que cambiar a mano u y es problemático
        plt.figtext(0.93, 0.83, 'E(k)',fontsize='xx-large',fontstyle='italic',fontfamily='serif')
        #ax.set_zticks([2,1,0,-1,-2])
        ax.invert_zaxis()
        ax.view_init(-170, 60)
        """
        Format for the plot (to edit)
        """
        plt.show()
        
    def toros(self,n_kpoints):
        """
        Function that plots half the torus defined by the d vector (in a half of the 2D Brillouin zone), to graphically determine the Chern number
        """
        k_step = 2.0*pi/n_kpoints
        kxpoints = arange(0,pi+k_step,k_step)
        kypoints = arange(-pi,pi+k_step,k_step)
        X,Y=meshgrid(kxpoints,kypoints)
        
        d=[sin(X), sin(Y), self.u_value+cos(X)+cos(Y)]

        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_zlim(-4,2)
        ax1.plot_surface(d[0], d[1], d[2], rstride=20, cstride=20, color='orange', edgecolors='orange')
        
        ax1.view_init(0, 130)

        plt.figtext(0.15, 0.36, 'd$_{x}$',fontsize='x-large',fontstyle='italic',fontfamily='serif')
        plt.figtext(0.38, 0.35, 'd$_{y}$',fontsize='x-large',fontstyle='italic',fontfamily='serif')
        plt.figtext(0.5, 0.49, 'd$_{z}$',fontsize='x-large',fontstyle='italic',fontfamily='serif')
        plt.figtext(0.15, 0.63, 'u=-1.5',fontsize='x-large',fontstyle='italic',fontfamily='serif') #hay que cambiar a mano u y es problemático
        ax1.scatter(0,0,0,color='black',marker='.')
        plt.show()
        
        
'''
Now, with this created, we only have to write in the lines below, or the console, what part of the program we want to run exactly

Something like: Ribbon(100).plot_bands(200,2)
'''