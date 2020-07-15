
# coding: utf-8

# # Homework Assignment 17
# 
# 
# ## Instructions
# 
# Consider the reservoir shown below with the given properties that has been discretized into equal grid blocks.
# 
# ![image](images/grid.png)
# 
# To be clear, there is a constant-rate injector of 1000 ft$^3$/day at $x$ = 5000 ft, $y$ = 5000 ft and a constant BHP well (producer) with $p_w$ = 800 psi at $x$ = 9000 ft, $y$ = 9000 ft. Both wells have a radius of 0.25 ft and no skin factor.
# 
# Use the code you wrote in [Assignment 15](https://github.com/PGE323M-Fall2017/assignment15) and add additional functionality to incorporate the wells.  The wells section of the inputs will look something like:
# 
# ```python
# inputs['wells'] = {
#             'rate': {
#                 'locations': [(0.0, 1.0), (2.0, 2.0)],
#                 'values': [1000, 1000],
#                 'radii': [0.25, 0.25]
#             },
#             'bhp': {
#                 'locations': [(6250.0, 1.0)],
#                 'values': [800],
#                 'radii': [0.25],
#                 'skin factor': 0.0
#             }
#         }
# ```
# 
# notice that all the values are Python lists so that multiple wells of each type can be included.  The `'locations'` keyword has a value that is a list of tuples.  Each tuple contains the $x,y$ Cartesian coordinate pair that gives the location of the well.  You must write some code that can take this $x,y$-pair and return the grid block number that the well resides in.  This should be general enough that changing the number of grids in the $x$ and $y$ directions still gives the correct grid block.  Once you know the grid block numbers for the wells, the changes to `fill_matrices()` should be relatively easy.
# 
# All of the old tests from the last few assignments are still in place, so your code must run in the abscense of any well section in your inputs.

# In[1]:

import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import yaml

from assignment13 import OneDimReservoir


# In[2]:

class TwoDimReservoir(OneDimReservoir):
    
    def __init__(self, inputs):
        '''
            Class for solving one-dimensional reservoir problems with
            finite differences.
        '''
        
        #stores input dictionary as class attribute
        if isinstance(inputs, str):
            with open(inputs) as f:
                self.inputs = yaml.load(f)
        else:
            self.inputs = inputs
        
        #assigns class attributes from input data
        self.parse_inputs()
        
        #calls fill matrix method (must be completely implemented to work)
        self.fill_matrices()
        
        #applies the initial reservoir pressues to self.p
        self.apply_initial_conditions()
        
        #create an empty list for storing data if plots are requested
        if 'plots' in self.inputs:
            self.p_plot = []
            
        return
    
    def parse_inputs(self):
        
        self.Vis = self.inputs['fluid']['viscosity']
        self.Bo = self.inputs['fluid']['formation volume factor']
        self.Nx = self.inputs['numerical']['number of grids']['x']
        self.Ny = self.inputs['numerical']['number of grids']['y']
        self.length = self.inputs['reservoir']['length']
        self.Co = self.inputs['fluid']['compressibility']
        self.N = self.Nx*self.Ny
    
        if 'conversion factor' in self.inputs:
            self.C_fac = self.inputs['conversion factor']
            
        else:
            self.C_fac = 1
                
        # k A and porosity
        phi = self.inputs['reservoir']['porosity']
        perm = self.inputs['reservoir']['permeability']
        #area = self.inputs['reservoir']['cross sectional area']
        depth = self.inputs['reservoir']['depth']
        
        self.phi = self.input_and_return_data(phi)
        self.perm = self.input_and_return_data(perm)
        #self.area = self.input_and_return_data(area)
        self.depth = self.input_and_return_data(depth)
        # put if conversion factor to make generic code
        
        self.delta_x = self.assign_delta_x_array()
        self.delta_y = self.assign_delta_y_array()
        
    def assign_delta_x_array(self):
        
        if 'delta x' not in self.inputs['numerical']:
            length = self.inputs['reservoir']['length']
            delta_x = np.float(length)/self.Nx
            delta_x_arr = np.ones(self.N) * delta_x
            
        else:
            delta_x_arr = np.array(self.inputs['numerical']['delta x'],dtype=np.double)
            length_delta_x_arr = delta_x_arr.shape[0]
            
        return delta_x_arr
            
            #assert length_delta_x_arr == self.N,(" User defined delta x does not match with number of grids ")
    
    def assign_delta_y_array(self):
        
        if 'delta y' not in self.inputs['numerical']:
            height = self.inputs['reservoir']['height']
            delta_y = np.float(height)/self.Ny
            delta_y_arr = np.ones(self.N) * delta_y
            
        else:
            delta_y_arr = np.array(self.inputs['numerical']['delta y'],dtype=np.double)
            length_delta_y_arr = delta_y_arr.shape[0]
            
        return delta_y_arr
            
            #assert length_delta_x_arr == self.N,(" User defined delta x does not match with number of grids ")
    
    
    def input_and_return_data(self,input_data):
        '''
            To check the input a tuple or a single value
        '''
        N = self.N
        # Start Here
        # use isinstance
        if isinstance(input_data,(list,tuple)):
            output = input_data
        elif isinstance(input_data,str):
            output = np.loadtxt(input_data)
        else:
            output = input_data* np.ones(N)
        
        return output
            
    
    def compute_transmissibility(self, i, j):
        '''
            Computes the transmissibility.
        '''
        B = self.Bo
        V = self.Vis
        d = self.depth
        x = self.delta_x
        y = self.delta_y
        k = self.perm
        
        if k[i] == 0 and k[j] ==0:
            T = 0
        else:    
            if abs(j-i) < self.Nx:
                kx = k[i]*k[j]*((x[i]+x[j])/((x[i]*k[j])+(x[j]*k[i])))
                T = (1/(B*V))*(kx*y[i]*d[i]/((x[i]+x[j])/2))
            else:
                ky = k[i]*k[j]*((y[i]+y[j])/((y[i]*k[j])+(y[j]*k[i])))
                T = (1/(B*V))*(ky*x[i]*d[i]/((y[i]+y[j])/2))
        return T 
    
    
    
    def compute_accumulation(self, i):
        '''
            Computes the accumulation.
        '''
        B = self.Bo
        C = self.Co
        p = self.phi
        d = self.depth
        x = self.delta_x
        y = self.delta_y
        
        c1 = (p[i]*C)/B
        c2 = d[i]*x[i]*y[i]
        
        return c1*c2
    
        
    def fill_matrices(self):
        '''
           Assemble the transmisibility, accumulation matrices, and the flux
           vector.  Returns sparse data-structures
        '''
        
        Nx = self.Nx
        Ny = self.Ny
        C = self.C_fac
        Co = self.Co
        Bo = self.Bo
        d = self.depth
        k = self.perm
        V = self.Vis
        
        N = Nx * Ny
        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']
        
        # Wells Location
        
        dx = np.average(self.delta_x)
        dy = np.average(self.delta_y)
        list_pressure = []
        list_rate = []
        
        if 'wells' in self.inputs:
            for x,y in self.inputs['wells']['rate']['locations']:
                x1 = int(x/dx)
                y1 = int(y/dy)
                list_rate += [(y1*Nx)+x1]
            for X,Y in self.inputs['wells']['bhp']['locations']:
                X1 = int(X/dx)
                Y1 = int(Y/dy)
                list_pressure += [(Y1*Nx)+X1]
        
        # B,T,Q Matrix
        Q = np.zeros(N)
        B = scipy.sparse.identity(N,format='csc')
        for i in range(0,N):
            B[i,i] = self.compute_accumulation(i)
            
        T = lil_matrix((N,N))
        for i in range(N):
            if Ny == 1:
                if i == 0:
                    if self.inputs['boundary conditions']['left']['type'] == 'prescribed pressure':
                        T[i,i] = (2*self.compute_transmissibility(i,i) + self.compute_transmissibility(i,i+1))*C
                        T[i,i+1] = -self.compute_transmissibility(i,i+1)*C
                        Q[i] = 2*self.compute_transmissibility(i,i)*C*self.inputs['boundary conditions']['left']['value']
                    else:
                        T[i,i] = (0 + self.compute_transmissibility(i,i+1))*C
                        T[i,i+1] = -self.compute_transmissibility(i,i+1)*C
                        Q[i] = 0
                elif i == N-1:
                    if self.inputs['boundary conditions']['right']['type'] == 'prescribed pressure':
                        T[i,i] = (2*self.compute_transmissibility(i,i) + self.compute_transmissibility(i,i-1))*C
                        T[i,i-1] = -self.compute_transmissibility(i,i-1)*C
                        Q[i] = 2*self.compute_transmissibility(i,i)*C*self.inputs['boundary conditions']['right']['value']
                    else:
                        T[i,i] = (0 + self.compute_transmissibility(i,i-1))*C
                        T[i,i-1] = -self.compute_transmissibility(i,i-1)*C
                        Q[i] = 0
                else:
                    T[i,i] = (self.compute_transmissibility(i,i-1)+self.compute_transmissibility(i,i+1))*C
                    T[i,i-1] = -self.compute_transmissibility(i,i-1)*C
                    T[i,i+1] = -self.compute_transmissibility(i,i+1)*C
            
            else:
                ## if i is not on bottom edge of grid
                if i > (Nx-1):
                    T[i,i-Nx]= -self.compute_transmissibility(i,i-Nx)*C
                    #T[i,i] = T[i,i] + (self.compute_transmissibility(i,i-Nx)*C)

                ## if i is not on top edge of grid
                if i < ((Nx*Ny)-Nx):
                    T[i,i+Nx]= -self.compute_transmissibility(i,i+Nx)*C
                    #T[i,i] = T[i,i] + (self.compute_transmissibility(i,i+Nx)*C)

                ## if i is not on left edge of grid
                if (i%Nx) != 0 and i-1>=0:
                    if i%Nx == Nx-1:
                        if self.inputs['boundary conditions']['right']['type'] == 'prescribed pressure':
                            T[i,i-1]= -self.compute_transmissibility(i,i-1)*C
                            T[i,i] = -(2*self.compute_transmissibility(i,i)*C)
                            Q[i]=2*self.compute_transmissibility(i,i)*C*self.inputs['boundary conditions']['right']['value']
                        else:
                            T[i,i-1]= -self.compute_transmissibility(i,i-1)*C
                            T[i,i] = 0
                            Q[i]= 0
                    else:
                        T[i,i-1]= -(self.compute_transmissibility(i,i-1))*C
                        T[i,i] = 0

                ## if i is not on right edge of grid
                if (i+1)%Nx != 0 and (i+1)<=((Nx*Ny)-1):
                    
                    T[i,i+1]= -(self.compute_transmissibility(i,i+1))*C 
                
                T[i,i]= -np.sum(T[i])
        
        
        Np = len(list_pressure)
        Nq = len(list_rate)
        if 'wells' in self.inputs:
            r_pressure =self.inputs['wells']['bhp']['radii']
            r_rate = self.inputs['wells']['rate']['radii']
            rq = 0
            for i in range(Np):
                A = list_pressure[i]
                rq = 0.14 * np.sqrt(self.delta_x[A]**2+self.delta_y[A]**2)
                J = 2*C*np.pi*d[A]*k[A]/(V*Bo*np.log(rq/r_pressure[i]))
                T[A,A] = T[A,A]+J
                Q[A] = Q[A] + (J * self.inputs['wells']['bhp']['values'][i])
            for i in range(Nq):
                a = list_rate[i]
                Q[a] = Q[a] + self.inputs['wells']['rate']['values'][i]
        
    
        
        self.T = T.tocsr()
        self.Q = Q
        self.B = B
        return      
    
    def apply_initial_conditions(self):
        '''
            Applies initial pressures to self.p
        '''
        N = self.Nx*self.Ny
        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']
        return


#     
#                     
# '''
# if i%Nx == 0:
#     if self.inputs['boundary conditions']['left']['type'] == 'prescribed pressure':
#         T[i,i+1]= -(self.compute_transmissibility(i,i+1))*C
#         T[i,i] = -(2*self.compute_transmissibility(i,i)*C)
#         Q[i]=2*self.compute_transmissibility(i,i)*C*self.inputs['boundary conditions']['left']['value']
#     else:
#         T[i,i+1]= -(self.compute_transmissibility(i,i+1))*C
#         T[i,i] = 0
#         Q[i]= 0
# else:
#     T[i,i+1]= -(self.compute_transmissibility(i,i+1))*C 
#     T[i,i] = 0
# '''
# 
