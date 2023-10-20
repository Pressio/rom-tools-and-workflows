'''
The scaler class is used to performed scaled POD.

*What is scaled POD, and why would I do it?*

Standard POD computes a basis that minimizes the projection error in a standard Euclidean $\\ell^2$ inner product,
i.e., for a snapshot matrix $\\mathbf{S}$, POD computes the basis by solving the minimization problem
(assuming no affine offset)
$$ \\boldsymbol \\Phi = \\underset{ \\boldsymbol \\Phi_{\\*} \\in \\mathbb{R}^{N \\times K} | \\boldsymbol
\\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}}{ \\mathrm{arg \\; min} } \\| \\Phi_{\\*} \\Phi_{\\*}^T
\\mathbf{S} - \\mathbf{S} \\|_2.$$
In this minimization problem, errors are measured in a standard $\\ell^2$ norm.
For most practical applications, where our snapshot matrix involves variables of different scales,
this norm does not make sense (both intuitively, and on dimensional grounds).
As a practical example, consider fluid dynamics where the total energy is orders of magnitude larger than the density.

One of the most common approaches for mitigating this issue is to perform scaled POD.
In scaled POD, we solve a minimization problem on a scaled snapshot matrix.
Defining $\\mathbf{S}_{\\*} = \\mathbf{W}^{-1} \\mathbf{S}$, where $\\mathbf{W}$ is a weighting matrix
(e.g., a diagonal matrix containing the max absolute value of each state variable),
we compute the basis as the solution to the minimization problem
$$ \\boldsymbol \\Phi = \\mathbf{W} \\underset{ \\boldsymbol \\Phi_{\\*} \\in \\mathbb{R}^{N \\times K} |\\boldsymbol
\\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}}{ \\mathrm{arg \\; min} } \\| \\Phi_{\\*} \\Phi_{\\*}^T
\\mathbf{S}_{\\*} - \\mathbf{S}_{\\*} \\|_2.$$

The Scaler encapsulates this information
'''
import numpy as np
import scipy
import abc
class AbstractScaler(abc.ABC):
    '''
    Abstract base class
    '''
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def preScaling(self, my_array: np.ndarray) -> np.ndarray:
        '''
        Scales the snapshot matrix before performing SVD
        '''
        pass

    @abc.abstractmethod
    def postScaling(self, my_array: np.ndarray) -> np.ndarray:
        '''
        Scales the left singular vectors after performing SVD
        '''
        pass


##================
#Helper functions
#=================
def extractIthVariableData(i,n_var,data_matrix,variable_ordering):
    if variable_ordering == 'F':
        return data_matrix[i::n_var]
    elif variable_ordering == 'C':
        nx = int( data_matrix.shape[0] / n_var)
        start_index = i*nx
        end_index = (i+1)*nx
        return data_matrix[start_index:end_index]

def scaleDataMatrixForIthVar(i,n_var,data_matrix,variable_ordering,var_scale):
    if variable_ordering == 'F':
        data_matrix[i::n_var] *= var_scale
    elif variable_ordering == 'C':
        nx = int( data_matrix.shape[0] / n_var)
        start_index = i*nx
        end_index = (i+1)*nx
        data_matrix[start_index:end_index] *= var_scale
    return data_matrix


class NoOpScaler(AbstractScaler):
    '''
    No op implementation
    '''
    def __init__(self):
        pass

    def preScaling(self,data_matrix):
        return data_matrix

    def postScaling(self,data_matrix):
        return data_matrix


class VectorScaler(AbstractScaler):
    '''
    Concrete implementation designed to scale snapshot matrices by a vector.
    For a snapshot matrix $\\mathbf{S} \\in \\mathbb{R}^{N \\times K}$, the VectorScaler
    accepts in a scaling vector $\\mathbf{v} \\in \\mathbb{R}^N$, and scales by
    $$\\mathbf{S}^* = \\mathrm{diag}(\\mathbf{v})^{-1} \\mathbf{S}$$
    before performing POD (i.e., POD is performed on $\\mathbf{S}^*$). After POD is performed, the bases
    are post-scaled by $$\\boldsymbol \\Phi = \\mathrm{diag}(\\mathbf{v}) \\mathbf{U}$$

    **Note that scaling can cause bases to not be orthonormal; we do not recommend using scalers with
    the NoOpOrthonormalizer**
    '''
    def __init__(self,scaling_vector):
        '''
        Constructor for the VectorScaler.

        Args:
            scaling_vector: Array containing the scaling vector for each row in the snapshot matrix.

        This constructor initializes the VectorScaler with the specified scaling vector.
        '''
        self.__scaling_vector_matrix = scipy.sparse.diags(scaling_vector)
        self.__scaling_vector_matrix_inv = scipy.sparse.diags(1./scaling_vector)


    def preScaling(self, data_matrix):
        '''
        Scales the input data matrix using the inverse of the scaling vector and returns the scaled matrix.

        Args:
            data_matrix (np.ndarray): The input data matrix to be scaled.

        Returns:
            np.ndarray: The scaled data matrix.
        '''
        return self.__scaling_vector_matrix_inv @ data_matrix

    def postScaling(self,data_matrix):
        '''
        Scales the input data matrix using the scaling vector and returns the scaled matrix.

        Args:
            data_matrix (np.ndarray): The input data matrix to be scaled.

        Returns:
            np.ndarray: The scaled data matrix.
        '''
        return self.__scaling_vector_matrix  @ data_matrix



#=========
#This class is designed to scale a data matrix comprising multiple states
#(e.g., for the Navier--Stokes, rho, rho u, rhoE)
#========
class VariableScaler(AbstractScaler):
    '''
    Concrete implementation designed for snapshot matrices involving multiple state variables.
    This scaler will scales each variable based on
      - max-abs scaling: for the $i$th state variable $u_i$, we will compute the scaling as
        $s_i = \\mathrm{max}( \\mathrm{abs}( S_i ) )$, where $S_i$ denotes the snapshot matrix of the $i$th variable.
      - mean abs: for the $i$th state variable $u_i$, we will compute the scaling as
        $s_i = \\mathrm{mean}( \\mathrm{abs}( S_i ) )$, where $S_i$ denotes the snapshot matrix of the $i$th variable.
      - variance: for the $i$th state variable $u_i$, we will compute the scaling as
        $s_i = \\mathrm{std}( S_i ) $, where $S_i$ denotes the snapshot matrix of the $i$th variable.

    **This class requires you to specify variable ordering (either 'F' or 'C')**
    For a state with variables $u,v,w$ defined at $n$ discrete points, these orderings are

    **Order F variable ordering: $[u_1,v_1,w_1,u_2,v_2,w_2,...,u_n,v_n,w_n]$**

    **Order C variable ordering: $[u_1,u_2,...,u_n,v_1,...]$**
    '''
    def __init__(self,scaling_type,variable_ordering,n_var):
        '''
        Constructor for the VariableScaler.

        Args:
            scaling_type (str): The scaling method to use ('max_abs', 'mean_abs', or 'variance').
            variable_ordering (str): Variable ordering ('F' or 'C').
            n_var (int): The number of state variables.

        This constructor initializes the VariableScaler with the specified scaling type, variable ordering, and
        number of variables.
        '''
        assert variable_ordering == 'C' or variable_ordering == 'F', "Invalid variable ordering, options are F and C"
        self.__scaling_type = scaling_type
        self.__variable_ordering = variable_ordering
        self.have_scales_been_initialized = False
        self.__n_var = n_var

    def initializeScalings(self,data_matrix):
        '''
        Initializes the scaling factors for each state variable based on the specified method.

        Args:
            data_matrix (np.ndarray): The input data matrix.
        '''
        self.var_scales_ = np.ones(self.__n_var)
        for i in range(self.__n_var):
            ith_var = extractIthVariableData(i,self.__n_var,data_matrix,self.__variable_ordering)
            if self.__scaling_type == 'max_abs':
                var_scale = np.max(abs(ith_var))
            elif self.__scaling_type == 'mean_abs':
                var_scale = np.mean(abs(ith_var))
            elif self.__scaling_type == 'variance':
                var_scale = np.sqrt(np.var(ith_var))

            ## in case of a zero field (e.g., 2D)
            if var_scale < 1e-10:
                var_scale = 1.
            self.var_scales_[i] = var_scale
        self.have_scales_been_initialized = True

    ## These are all inplace operations
    def preScaling(self, data_matrix):
        '''
        Scales the input data matrix before processing, taking into account the previously initialized scaling factors.

        Args:
            data_matrix (np.ndarray): The input data matrix to be scaled.

        Returns:
            np.ndarray: The scaled data matrix.
        '''
        if self.have_scales_been_initialized:
            pass
        else:
            self.initializeScalings(data_matrix)
        # scale each field (variable scaling)
        for i in range(self.__n_var):
            data_matrix = scaleDataMatrixForIthVar(i,self.__n_var,data_matrix,self.__variable_ordering,1./self.var_scales_[i])
        return data_matrix

    def postScaling(self,data_matrix):
        assert self.have_scales_been_initialized, "Scales in VariableScaler have not been initialized"
        # scale each field
        for i in range(self.__n_var):
            data_matrix = scaleDataMatrixForIthVar(i,self.__n_var,data_matrix,self.__variable_ordering,self.var_scales_[i])
        return data_matrix

class VariableAndVectorScaler(AbstractScaler):
    '''
    Concrete implementation designed to scale snapshot matrices involving multiple state variables by both the variable
    magnitudes and an additional vector.
    This is particularly useful when wishing to perform POD for, e.g., a finite volume method where we want to scale by
    the cell volumes as well as the variable magnitudes. This implementation combines the VectorScaler and
    VariableScaler classes.
    '''

    def __init__(self,scaling_vector,scaling_type,variable_ordering,n_var):
        '''
        Constructor for the VariableAndVectorScaler.

        Args:
            scaling_vector: Array containing the scaling vector for each row in the snapshot matrix.
            scaling_type: Scaling method ('max_abs', 'mean_abs', or 'variance') for variable magnitudes.
            variable_ordering: Variable ordering ('F' or 'C') for state variables.
            n_var: The number of state variables.

        This constructor initializes the `VariableAndVectorScaler` with the specified parameters.
        '''
        self.__myVariableScaler = VariableScaler(scaling_type,variable_ordering,n_var)
        self.__myVectorScaler =  VectorScaler(scaling_vector)

    def preScaling(self, data_matrix):
        '''
        Scales the input data matrix before processing, first using the `VariableScaler` and then the `VectorScaler`.

        Args:
            data_matrix (np.ndarray): The input data matrix to be scaled.

        Returns:
            np.ndarray: The scaled data matrix.
        '''
        data_matrix = self.__myVariableScaler.preScaling(data_matrix)
        return self.__myVectorScaler.preScaling(data_matrix)

    def postScaling(self,data_matrix):
        '''
        Scales the input data matrix after processing, first using the `VectorScaler` and then the `VariableScaler`.

        Args:
            data_matrix (np.ndarray): The input data matrix to be scaled.

        Returns:
            np.ndarray: The scaled data matrix.
        '''
        data_matrix = self.__myVectorScaler.postScaling(data_matrix)
        return self.__myVariableScaler.postScaling(data_matrix)
