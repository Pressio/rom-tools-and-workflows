'''
The scaler class is used to performed scaled POD.

*What is scaled POD, and why would I do it?*

Standard POD computes a basis that minimizes the projection error in a standard Euclidean $\\ell^2$ inner product, i.e.,
for a snapshot matrix $\\mathbf{S}$, POD computes the basis by solving the minimization problem (assuming no affine offset)
$$ \\boldsymbol \\Phi = \\underset{ \\boldsymbol \\Phi_{\\*} \\in \\mathbb{R}^{N \\times K} | \\boldsymbol \\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}}{ \\mathrm{arg \\; min} } \\| \\Phi_{\\*} \\Phi_{\\*}^T \\mathbf{S} - \\mathbf{S} \\|_2.$$
In this minimization problem, errors are measured in a standard $\\ell^2$ norm.
For most practical applications, where our snapshot matrix involves variables of different scales,
this norm does not make sense (both intuitively, and on dimensional grounds).
As a practical example, consider fluid dynamics where the total energy is orders of magnitude larger than the density.

One of the most common approaches for mitigating this issue is to perform scaled POD.
In scaled POD, we solve a minimization problem on a scaled snapshot matrix.
Defining $\\mathbf{S}_{\\*} = \\mathbf{W}^{-1} \\mathbf{S}$, where $\\mathbf{W}$ is a weighting matrix
(e.g., a diagonal matrix containing the max absolute value of each state variable),
we compute the basis as the solution to the minimization problem
$$ \\boldsymbol \\Phi = \\mathbf{W} \\underset{ \\boldsymbol \\Phi_{\\*} \\in \\mathbb{R}^{N \\times K} | \\boldsymbol \\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}}{ \\mathrm{arg \\; min} } \\| \\Phi_{\\*} \\Phi_{\\*}^T \\mathbf{S}_{\\*} - \\mathbf{S}_{\\*} \\|_2.$$

The Scaler encapsulates this information
'''
import numpy as np
import scipy
import abc
class AbstractScaler(abc.ABC):
    """
    Abstract base class
    """
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def preScaling(self, my_array: np.ndarray) -> np.ndarray:
        """
        Scales the snapshot matrix before performing SVD
        """
        pass

    @abc.abstractmethod
    def postScaling(self, my_array: np.ndarray) -> np.ndarray:
        """
        Scales the left singular vectors after performing SVD
        """
        pass



class NoOpScaler(AbstractScaler):
    """
    No op implementation
    """
    def __init__(self):
        pass

    def preScaling(self,data_matrix):
        return data_matrix

    def postScaling(self,data_matrix):
        return data_matrix


class VectorScaler(AbstractScaler):
    """
    Concrete implementation designed to scale snapshot matrices by a vector.
    For a snapshot tensor $\\mathbf{S} \\in \\mathbb{R}^{N_{\\mathrm{u}} \\times N \\times K}$, the VectorScaler
    accepts in a scaling vector $\\mathbf{v} \\in \\mathbb{R}^{N$, and scales by
    $$\\mathbf{S}^* = \\mathrm{diag}(\\mathbf{v})^{-1} \\mathbf{S}$$
    before performing POD (i.e., POD is performed on $\\mathbf{S}^*$). After POD is performed, the bases
    are post-scaled by $$\\boldsymbol \\Phi = \\mathrm{diag}(\\mathbf{v}) \\mathbf{U}$$

    **Note that scaling can cause bases to not be orthonormal; we do not recommend using scalers with the NoOpOrthonormalizer**
    """
    def __init__(self,scaling_vector):
        '''
        Inputs: scaling_vector: array containing the scaling vector for each row in the snapshot matrix
        '''
        self.__scaling_vector_matrix = scaling_vector 
        self.__scaling_vector_matrix_inv = 1./scaling_vector


    def preScaling(self, data_matrix):
        return self.__scaling_vector_matrix_inv[None,:,None] * data_matrix

    def postScaling(self,data_matrix):
        return self.__scaling_vector_matrix[None,:,None] * data_matrix



#=========
#This class is designed to scale a data matrix comprising multiple states
#(e.g., for the Navier--Stokes, rho, rho u, rhoE)
#========
class VariableScaler(AbstractScaler):
    """
    Concrete implementation designed for snapshot matrices involving multiple state variables.
    This scaler will scales each variable based on
      - max-abs scaling: for the $i$th state variable $u_i$, we will compute the scaling as $s_i = \\mathrm{max}( \\mathrm{abs}( S_i ) )$, where $S_i$ denotes the snapshot matrix of the $i$th variable.
      - mean abs: for the $i$th state variable $u_i$, we will compute the scaling as $s_i = \\mathrm{mean}( \\mathrm{abs}( S_i ) )$, where $S_i$ denotes the snapshot matrix of the $i$th variable.
      - variance: for the $i$th state variable $u_i$, we will compute the scaling as $s_i = \\mathrm{std}( S_i ) $, where $S_i$ denotes the snapshot matrix of the $i$th variable.

    **This class requires you to specify variable ordering (either 'F' or 'C')**
    For a state with variables $u,v,w$ defined at $n$ discrete points, these orderings are

    """
    def __init__(self,scaling_type):
        self.__scaling_type = scaling_type
        self.have_scales_been_initialized = False

    def initializeScalings(self,data_tensor):
        n_var = data_tensor.shape[0]
        self.var_scales_ = np.ones(n_var)
        for i in range(n_var):
            ith_var = data_tensor[i] 
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
    def preScaling(self, data_tensor):
        n_var = data_tensor.shape[0]
        if self.have_scales_been_initialized:
            pass
        else:
            self.initializeScalings(data_tensor)
        # scale each field (variable scaling)
        for i in range(n_var):
            data_tensor[i] = data_tensor[i] / self.var_scales_[i]
        return data_tensor

    def postScaling(self,data_tensor):
        assert self.have_scales_been_initialized, "Scales in VariableScaler have not been initialized"
        # scale each field
        n_var = data_tensor.shape[0]
        for i in range(n_var):
            data_tensor[i] = data_tensor[i]*self.var_scales_[i]
        return data_tensor

class VariableAndVectorScaler(AbstractScaler):
    """
    Concrete implementation designed to scale snapshot matrices involving multiple state variables by both the variable magnitudes and an additional vector.
    This is particularly useful when wishing to perform POD for, e.g., a finite volume method where we want to scale by the cell volumes as well as the variable
    magnitudes. This implementation combines the VectorScaler and VariableScaler classes
    """

    def __init__(self,scaling_vector,scaling_type):
        '''
        Inputs: scaling_vector: array containing the scaling vector for each row in the snapshot matrix
        scaling_type: 'max_abs','mean_abs', or 'variance'
        '''
        self.__myVariableScaler = VariableScaler(scaling_type)
        self.__myVectorScaler =  VectorScaler(scaling_vector)

    def preScaling(self, data_tensor):
        data_tensor = self.__myVariableScaler.preScaling(data_tensor)
        return self.__myVectorScaler.preScaling(data_tensor)

    def postScaling(self,data_tensor):
        data_tensor = self.__myVectorScaler.postScaling(data_tensor)
        return self.__myVariableScaler.postScaling(data_tensor)
