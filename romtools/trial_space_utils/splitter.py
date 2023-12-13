'''
The Splitter class is used to "split" a snapshot matrix before constructing the trial space.

*What do we mean by splitting a trial space, and why do it?*
Trial space splitting can be important for vector-valued equations where there are multiple state variables. As an example, we could consider the compressible Euler equations in one dimension in which case the spatially continuous (conserved) variable set is
$$\\boldsymbol u(x) = \\begin{bmatrix} \\rho(x) &  \\rho u (x)& \\rho E(x) \\end{bmatrix}^T.$$
There are several ways of constructing ROMs of vector-valued systems. In one approach, we simply stack all of our degrees of freedom and create a global basis. Mathematically, this comprises constructing a basis function that looks as follows
$$
\\boldsymbol \\Phi(x) = \\begin{bmatrix}
\\Phi_{1}^{\\rho}(x) & \\Phi_{2}^{\\rho }(x) & \\cdots & \\Phi_{K}^{\\rho}(x) \\\\\\\\
\\Phi_{1}^{\\rho u}(x) & \\Phi_{2}^{\\rho u }(x) & \\cdots & \\Phi_{K}^{\\rho u}(x) \\\\\\
\\Phi_{1}^{\\rho E}(x) & \\Phi_{2}^{\\rho E }(x) & \\cdots & \\Phi_{K}^{\\rho E}(x) \\
\\end{bmatrix},
$$
and the state is approximated by
$$\\tilde{\\boldsymbol u}(x) = \\sum_{i=1}^{K} \\Phi_i(x) \\hat{\\boldsymbol u}(t).$$
With this representation, the different state variables share generalized coordinates $\\hat{\\boldsymbol u}(t)$.

An alternative approach is to ensure that each state variable is linearly independent of the other by employing a block basis. With this approach, this comprises a basis function that looks like
$$
\\boldsymbol \\Phi(x) = \\begin{bmatrix}
\\Phi_{1}^{\\rho}(x) &  0                     &  0  & \\Phi_{2}^{\\rho }(x) & \\cdots \\\\\\\\
 0                   & \\Phi_{1}^{\\rho u }(x) &  0 & 0 & \\cdots \\\\\\
 0                   & 0                      & \\Phi_{1}^{\\rho E}(x) & 0 & \\cdots \\
\\end{bmatrix}.
$$
This "split" representation can be particularly advantageous for problems where different state variables span numerous orders of magnitude.

The splitting class encapsulates this functionality. We note that the splitter is **applied to the snapshot matrix before POD is performed**.

'''
from typing import Tuple
import abc
import numpy as np

class AbstractSplitter(abc.ABC):
    """Abstract class for a splitter"""
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, my_array: np.ndarray) -> np.ndarray:
        pass


class NoOpSplitter(AbstractSplitter):
    """Concrete no-op implementation"""
    def __init__(self) -> None:
        pass

    def __call__(self, my_array: np.ndarray):
        return my_array




class BlockSplitter(AbstractSplitter):
    """
    Splits a data matrix into blocks defined by a list, e.g., for our Euler equation example above, we could set blocks = [[0,1],[2]] which
    would result in
    $$
    \\boldsymbol \\Phi(x) = \\begin{bmatrix}
    \\Phi_{1}^{\\rho}(x)   &  0 & \\Phi_{2}^{\\rho}(x) &  0 & \\cdots  \\\\\\\\
    \\Phi_{1}^{\\rho u}(x) &  0 & \\Phi_{2}^{\\rho u}(x) & 0 & \\cdots \\\\\\
    0                      & \\Phi_{1}^{\\rho E }(x) & 0 & \\Phi_{2}^{\\rho E }(x) & \\cdots \\
    \\end{bmatrix}.
    $$
    """
    def __init__(self,blocks : list,n_var : int, variable_ordering = 'C'):
        assert variable_ordering == 'C' or variable_ordering == 'F', "Invalid variable ordering, options are F and C"
        self.__variable_ordering = variable_ordering
        self.__n_var = n_var
        self.__n_blocks = len(blocks)
        self.__blocks = blocks
        ## Check that the block list is valid
        my_vars = np.zeros(0,dtype='int')
        for block in blocks:
            for var in block:
                my_vars = np.append(my_vars,var)
        my_vars.sort()
        assert np.allclose(my_vars,np.arange(0,self.__n_var,dtype='int')), "Invalid block input"

    def __call__(self, my_array: np.ndarray):
        d1 = my_array.shape[0]
        d2 = my_array.shape[1]
        n = int(d1/self.__n_var)
        new_d2 = d2*self.__n_blocks
        my_split_array = np.zeros((d1,new_d2))
        for block_counter, block in enumerate(self.__blocks):
            start_col = block_counter*d2
            end_col = (block_counter+1)*d2
            for var in block:
                if self.__variable_ordering == 'F':
                    my_split_array[var::self.__n_var,start_col:end_col] = getDataMatrixForIthVar(var,self.__n_var,my_array,self.__variable_ordering)
                elif self.__variable_ordering == 'C':
                    start_index = var*n
                    end_index = (var+1)*n
                    my_split_array[start_index:end_index,start_col:end_col] = getDataMatrixForIthVar(var,self.__n_var,my_array,self.__variable_ordering)
        return my_split_array


def getDataMatrixForIthVar(i,n_var,data_matrix,variable_ordering='C'):
    """helper function to split data"""
    if variable_ordering == 'F':
        return data_matrix[i::n_var]
    elif variable_ordering == 'C':
        n = int( data_matrix.shape[0] / n_var)
        start_index = i*n
        end_index = (i+1)*n
        return data_matrix[start_index:end_index]

