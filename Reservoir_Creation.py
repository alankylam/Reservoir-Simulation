
# coding: utf-8

# # Assignment 13
# 
# Consider the reservoir shown below with the given properties that has been discretized into 4 equal grid blocks.
# 
# ![image](images/grid.png)
# 
# Below is a skeleton of a Python class that can be used to solve for the pressures in the reservoir.  The class is actually written generally enough that it can account for an arbitrary number of grid blocks, but we will only test cases with 4.  The class takes a Python dictonary of input parameters as an initialization argument.  An example of a complete set of input parameters is shown in the `setup()` function of the tests below.
# 
# Several simple useful functions are already implemented, your task is to implement the functions `compute_transmisibility()`, `compute_accumulation()`, `fill_matrices()` and `solve_one_step()`.  `fill_matrices()` should correctly populate the $\mathbf{T}$, $\mathbf{B}$ matrices as well as the vector $\vec{Q}$.   These should also correctly account for the application of boundary conditions.  Only the boundary conditions shown in the figure will be tested, but in preparation for future assignments, you may wish to add the logic to the code such that arbitrary pressure/no flow boundary conditions can be applied to either side of the one-dimensional reservoir. You may need to use the `'conversion factor'` for the transmissibilities.  `solve_one_step()` should solve a single time step for either the explicit or implicit methods depending on which is specified in the input parameters. The $\vec{p}{}^{n+1}$ values should be stored in the class attribute `self.p`.  If this is implemented correctly, you will be able to then use the `solve()` function to solve the problem up to the `'number of time steps'` value in the input parameters.
# 
# This time, in preparation for solving much larger systems of equations in the future, use the `scipy.sparse` module to create sparse matrix data structures for $\mathbf{T}$ and $\mathbf{B}$.  The sparsity of the matrix $\mathbf{T}$ is tested, so please assign this matrix to a class attribute named exactly `T`.
# 
# Once you have the tests passing, you might like to experiment with viewing several plots with different time steps, explicit vs. implicit, number of grid blocks, etc.  To assist in giving you a feel for how they change the character of the approximate solution.  I have implemented a simple plot function that might help for this.

# In[1]:

import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
import scipy.sparse.linalg


# In[2]:

class OneDimReservoir():
    
    def __init__(self, inputs):
        '''
            Class for solving one-dimensional reservoir problems with
            finite differences.
        '''
        
        #stores input dictionary as class attribute
        self.inputs = inputs
        
        #computes delta_x
        #self.delta_x = self.inputs['reservoir']['length'] / float(self.inputs['numerical']['number of grids'])
        
        #gets delta_t from inputs
        self.delta_t = self.inputs['numerical']['time step']
        
        self.list_inputs()
        
        #calls fill matrix method (must be completely implemented to work)
        self.fill_matrices()
        
        #applies the initial reservoir pressues to self.p
        self.apply_initial_conditions()
        
        #create an empty list for storing data if plots are requested
        if 'plots' in self.inputs:
            self.p_plot = []
            
        return
    
    def list_inputs(self):
        '''
            Store inputs for the function
        '''
        
        self.Vis = self.inputs['fluid']['viscosity']
        self.Bo = self.inputs['fluid']['formation volume factor']
        self.N = self.inputs['numerical']['number of grids']
        self.length = self.inputs['reservoir']['length']
        self.Co = self.inputs['fluid']['compressibility']
        
        if 'conversion factor' in self.inputs:
            self.C_fac = self.inputs['conversion factor']
            
        else:
            self.C_fac = 1
                
        # k A and porosity
        phi = self.inputs['reservoir']['porosity']
        perm = self.inputs['reservoir']['permeability']
        area = self.inputs['reservoir']['cross sectional area']
        
        self.phi = self.input_and_return_data(phi)
        self.perm = self.input_and_return_data(perm)
        self.area = self.input_and_return_data(area)
        # put if conversion factor to make generic code
        
        self.delta_x = self.assign_delta_x_array()
        
    def assign_delta_x_array(self):
        
        if 'delta x' not in self.inputs['numerical']:
            length = self.inputs['reservoir']['length']
            delta_x = np.float(length)/self.N
            delta_x_arr = np.ones(self.N) * delta_x
            
        else:
            delta_x_arr = np.array(self.inputs['numerical']['delta x'],dtype=np.double)
            length_delta_x_arr = delta_x_arr.shape[0]
            
        return delta_x_arr
            
            #assert length_delta_x_arr == self.N,(" User defined delta x does not match with number of grids ")
    
    def input_and_return_data(self,input_data):
        '''
            To check the input a tuple or a single value
        '''
        
        # Start Here
        # use isinstance
        if isinstance(input_data,(list,tuple)):
            output = input_data
        else:
            output = input_data* np.ones(self.N)
        return output
            
        
    def compute_transmissibility(self,i,j):
        '''
            Computes the transmissibility.
        '''
        # Complete implementation here
        # in this case i+1 = j
        k = self.perm
        a = self.area
        delta_x = self.delta_x
        
        #volume = (self.area*self.length)/self.N
        #return self.perm * self.area / (self.Vis * self.Bo * (volume/self.area)) 
        
        constant = 1/(self.Bo*self.Vis)
        constant2 = (2*k[i]*a[i]*k[j]*a[j])/((k[i]*a[i]*delta_x[j])+(k[j]*a[j]*delta_x[i]))
        return constant * constant2
    
    
    def compute_accumulation(self,i):
        '''
            Computes the accumulation.
        '''
        
        # Complete implementation here
        
        #volume = ((self.area * self.length) / self.N)
        #return ((volume * self.phi * self.Co) / self.Bo)
        
        # Function
        
        k = self.perm
        a = self.area
        phi= self.phi
        delta_x = self.delta_x
        
        return a[i]*delta_x[i]*phi[i]*self.Co/self.Bo
    
    
    def fill_matrices(self):
        '''
            Fills the matrices A, I, and \vec{p}_B and applies boundary
            conditions.
        '''
        
        N = self.inputs['numerical']['number of grids']
        
        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']
        
        # T and B matrix
        # T is A matrix multiply by transmissibility
        # B is I matrix multiply by accumulation
        #Complete implementation here
        
        Q = np.zeros(N)
        C = self.C_fac
        
        #Start here
    
        # B matrix
        I = scipy.sparse.identity(N,format='csc')
        for i in range(0,N):
            I[i,i] = self.compute_accumulation(i)
           
        V = N
        T = lil_matrix((N,N))

        for i in range(0,N):
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

    
        # A , I , Pb completed
        self.B = I
        self.T = T.tocsr()
        self.Q = Q
        
        return
        
                
    def apply_initial_conditions(self):
        '''
            Applies initial pressures to self.p
        '''
        
        N = self.inputs['numerical']['number of grids']
        
        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']
        
        return
                
                
    def solve_one_step(self):
        '''
            Solve one time step using either the implicit or explicit method
        '''
        
        #Complete implementation here
        viscosity = self.inputs['fluid']['viscosity']
        time = self.inputs['numerical']['time step']
        delta_time = self.inputs['numerical']['time step']
        #self.Q = ((1/delta_time)*self.B).dot(self.Pb_vec)
        self.Q = self.Q
        #self.B_inv = scipy.sparse.linalg.inv(self.B)
        
          
        if self.inputs['numerical']['solver'] == 'explicit':
            
            self.p = self.p + ((delta_time * 1 / self.B.diagonal()) * (self.Q-((self.T.dot(self.p)))))
            
        elif self.inputs['numerical']['solver'] == 'implicit':
            
            self.p = scipy.sparse.linalg.cg((self.T + (self.B/delta_time)),(((self.B/delta_time).dot(self.p))+self.Q))[0]
            
        elif "mixed method" in self.inputs['numerical']['solver']:
            
            theta = self.inputs['numerical']['solver']['mixed method']['theta']   
            self.p = scipy.sparse.linalg.cg((((1-theta)*self.T) + (self.B/delta_time)),((((self.B/delta_time)-(theta*self.T)).dot(self.p))+self.Q))[0]
        
        
        return
                        
    def solve(self):
        '''
            Solves until "number of time steps"
        '''
        
        for i in range(self.inputs['numerical']['number of time steps']):
            self.solve_one_step()
            
            if i % self.inputs['plots']['frequency'] == 0:
                self.p_plot += [self.get_solution()]
                
        return
    
    def plot(self):
        '''
           Crude plotting function.  Plots pressure as a function of grid block #
        '''
        
        if self.p_plot is not None:
            for i in range(len(self.p_plot)):
                plt.plot(self.p_plot[i])
        
        return
            
    def get_solution(self):
        '''
            Returns solution vector
        '''
        # scipy.sparse.spsolve instead of np.linalg.solve
        return self.p


# TEST CODE

# In[3]:

def setup():
    
    inputs = {
        'conversion factor': 6.33e-3,
        'fluid': {
            'compressibility': 1e-6, #psi^{-1}
            'viscosity': 1, #cp
            'formation volume factor': 1 
        },
        'reservoir': {
            'permeability': 50, #mD
            'porosity': 0.2,
            'length': 10000, #ft
            'cross sectional area': 200000 #ft^2
        },
        'initial conditions': {
            'pressure': 1000 #psi
        },
        'boundary conditions': {
            'left': {
                'type': 'prescribed pressure',
                'value': 2000 #psi
            },
            'right': {
                'type': 'prescribed flux',
                'value': 0 #ft^3/day
            }
        },
        'numerical': {
            'solver': 'implicit',
            'number of grids': 4,
            'time step': 1, #day
            'number of time steps' : 3 
        },
        'plots': {
            'frequency': 1
        }
    }
    
    return inputs

def test_compute_transmissibility():
    
    parameters = setup()
    
    problem = OneDimReservoir(parameters)
    
    np.testing.assert_allclose(problem.compute_transmissibility(0,0), 4000.0)
    
    return

def test_compute_accumulation():
    
    parameters = setup()
    
    problem = OneDimReservoir(parameters)
    
    np.testing.assert_allclose(problem.compute_accumulation(0), 100.0)
    
    return 

def test_is_transmissiblity_matrix_sparse():
    
    parameters = setup()
    
    problem = OneDimReservoir(parameters)
    
    assert scipy.sparse.issparse(problem.T)
    
    return
  
def test_implicit_solve_one_step():
    
    parameters = setup()
    
    implicit = OneDimReservoir(parameters)
    implicit.solve_one_step()
    np.testing.assert_allclose(implicit.get_solution(), 
                               np.array([1295.1463, 1051.1036, 1008.8921, 1001.7998]), 
                               atol=0.5)
    return

def test_explicit_solve_one_step():
    
    parameters = setup()
    
    parameters['numerical']['solver'] = 'explicit'
    
    explicit = OneDimReservoir(parameters)
    
    explicit.solve_one_step()

    np.testing.assert_allclose(explicit.get_solution(), 
                           np.array([ 1506., 1000.,  1000.,  1000.004]), 
                           atol=0.5)
    return 

def test_mixed_method_solve_one_step_implicit():
    
    parameters = setup()
    
    parameters['numerical']['solver'] = {'mixed method': {'theta': 0.0}}
    
    mixed_implicit = OneDimReservoir(parameters)
    
    mixed_implicit.solve_one_step()

    np.testing.assert_allclose(mixed_implicit.get_solution(), 
                           np.array([1295.1463, 1051.1036, 1008.8921, 1001.7998]), 
                           atol=0.5)
    return 

def test_mixed_method_solve_one_step_explicit():
    
    parameters = setup()
    
    parameters['numerical']['solver'] = {'mixed method': {'theta': 1.0}}
    
    mixed_explicit = OneDimReservoir(parameters)
    
    mixed_explicit.solve_one_step()

    np.testing.assert_allclose(mixed_explicit.get_solution(), 
                           np.array([ 1506., 1000.,  1000.,  1000.004]), 
                           atol=0.5)
    return 

def test_mixed_method_solve_one_step_crank_nicolson():
    
    parameters = setup()
    
    parameters['numerical']['solver'] = {'mixed method': {'theta': 0.5}}
    
    mixed = OneDimReservoir(parameters)
    
    mixed.solve_one_step()
    
    np.testing.assert_allclose(mixed.get_solution(), 
                               np.array([ 1370.4,  1037.8 ,  1003.8,  1000.4]),
                               atol=0.5)
    return 

def test_implicit_solve():
    
    parameters = setup()
    
    implicit = OneDimReservoir(parameters)
    implicit.solve()
    np.testing.assert_allclose(implicit.get_solution(), 
                               np.array([1582.9, 1184.8, 1051.5, 1015.9]), 
                               atol=0.5)
    return

def test_implicit_solve_reverse_boundary_conditions():
    
    parameters = setup()
    
    parameters['boundary conditions'] = {
            'right': {
                'type': 'prescribed pressure',
                'value': 2000 #psi
            },
            'left': {
                'type': 'prescribed flux',
                'value': 0 #ft^3/day
            }
        }
    
    implicit = OneDimReservoir(parameters)
    implicit.solve()
    np.testing.assert_allclose(implicit.get_solution(), 
                               np.array([1015.9, 1051.5, 1184.8, 1582.9]), 
                               atol=0.5)
    return

def test_explicit_solve():
    
    parameters = setup()
    
    parameters['numerical']['solver'] = 'explicit'
    
    explicit = OneDimReservoir(parameters)
    
    explicit.solve()

    np.testing.assert_allclose(explicit.get_solution(), 
                           np.array([1689.8, 1222.3, 1032.4, 1000.0]), 
                           atol=0.5)
    return 

def test_mixed_method_solve_crank_nicolson():
    
    parameters = setup()
    
    parameters['numerical']['solver'] = {'mixed method': {'theta': 0.5}}
    
    mixed = OneDimReservoir(parameters)
    
    mixed.solve()
    
    np.testing.assert_allclose(mixed.get_solution(), 
                               np.array([ 1642.0,  1196.5,  1043.8,  1009.1]),
                               atol=0.5)
    return

def test_implicit_heterogeneous_permeability_solve_one_step():
    
    parameters = setup()
    
    parameters['reservoir']['permeability'] = [10., 100., 50., 20] 
    
    implicit = OneDimReservoir(parameters)
    implicit.solve_one_step()
    np.testing.assert_allclose(implicit.get_solution(), 
                               np.array([1085.3,  1005.8,  1001.3,  1000.1]), 
                               atol=0.5)
    return

def test_implicit_heterogeneous_permeability_and_grid_size_solve_one_step():
    
    parameters = setup()
    
    parameters['reservoir']['permeability'] = [10., 100., 50., 20] 
    parameters['numerical']['delta x'] = [2000., 3000., 1500., 3500]
    
    implicit = OneDimReservoir(parameters)
    implicit.solve_one_step()
    np.testing.assert_allclose(implicit.get_solution(), 
                               np.array([1123.0,  1008.5,  1003.1,  1000.2]), 
                               atol=0.5)
    return


def test_implicit_heterogeneous_permeability_and_grid_size_solve():
    
    parameters = setup()
    
    parameters['reservoir']['permeability'] = [10., 100., 50., 20] 
    parameters['numerical']['delta x'] = [2000., 3000., 1500., 3500]
    
    implicit = OneDimReservoir(parameters)
    implicit.solve()
    np.testing.assert_allclose(implicit.get_solution(), 
                               np.array([1295.6,  1039.1,  1019.9,  1002.5]), 
                               atol=0.5)
    return


# In[4]:

parameters = setup()

problem = OneDimReservoir(parameters)

problem.fill_matrices()



# In[5]:

test_compute_transmissibility()


# In[6]:

test_compute_accumulation()


# In[7]:

test_is_transmissiblity_matrix_sparse()


# In[8]:

test_implicit_solve_one_step()


# In[9]:

test_explicit_solve_one_step()


# In[10]:

test_mixed_method_solve_one_step_implicit()


# In[11]:

test_mixed_method_solve_one_step_explicit()


# In[12]:

test_mixed_method_solve_one_step_crank_nicolson()


# In[13]:

test_implicit_solve()


# In[14]:

test_implicit_heterogeneous_permeability_solve_one_step()


# In[15]:

test_implicit_heterogeneous_permeability_and_grid_size_solve_one_step()


# In[ ]:



