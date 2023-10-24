"""
The OrthogonalizerClass is used to orthogonalize a basis at the end of the construction of a trial space.
Specifically, given a basis
$$\\boldsymbol \\Phi \\in \\mathbb{R}^{N \\times K},$$
the orthogonalizer will compute a new, orthogonalized basis $\\boldsymbol \\Phi_{\\*}$
where
$$\\boldsymbol \\Phi_{\\*}^T \\mathbf{W} \\boldsymbol \\Phi_{\\*} = \\mathbf{I}.$$
In the above, $\\mathbf{W}$ is a weighting matrix (typically the cell volumes).
"""
import abc
import numpy as np
import scipy.sparse
class AbstractOrthogonalizer(abc.ABC):
    '''
    Abstract base class
    '''
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, my_array: np.ndarray) -> np.ndarray:
        pass


class NoOpOrthogonalizer(AbstractOrthogonalizer):
    '''
    No op class (doesn't do anything)
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, my_array: np.ndarray):
        return my_array


class EuclideanL2Orthogonalizer(AbstractOrthogonalizer):
    '''
    Orthogonalizes the basis in the standard Euclidean L2 inner product, i.e., the output basis will satisfy
    $$\\boldsymbol \\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}.$$
    '''
    def __init__(self,qrFnc=None):
        """
        Constructor
        Args:

            qrFnc: a callable to use for computing the QR decomposition.
                    IMPORTANT: must conform to the API of [np.linalg.qr](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html).
                    If `None`, internally we use `np.linalg.qr`.
                    Note: this is useful when you want to use a custom qr, for example when your snapshots are distributed with MPI,
                    or maybe you have a fancy qr function that you can use.

        """
        self.__qrPicked = np.linalg.qr if qrFnc == None else qrFnc

    def __call__(self, my_array: np.ndarray):
        my_array,_ = self.__qrPicked(my_array,mode='reduced')
        return my_array

class EuclideanVectorWeightedL2Orthogonalizer(AbstractOrthogonalizer):
    '''
    Orthogonalizes the basis in vector-weighted Euclidean L2 inner product, i.e., the output basis will satisfy
    $$\\boldsymbol \\Phi_{\\*}^T \\mathrm{diag}(\\mathbf{w})\\boldsymbol \\Phi_{\\*} = \\mathbf{I},$$
    where $\\mathbf{w}$ is the weighting vector. Typically, this inner product is used for orthogonalizing with respect to cell volumes
    '''
    def __init__(self,weighting_vector: np.ndarray,qrFnc=None):
        """
        Constructor
        Args:
            weighting_vector: a 1-D numpy array that the matrix will be orthogonalized against. The length of the array must match the number of rows in the matrix that will be orthogonalized.
            qrFnc: a callable to use for computing the QR decomposition.
                    IMPORTANT: must conform to the API of [np.linalg.qr](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html).
                    If `None`, internally we use `np.linalg.qr`.
                    Note: this is useful when you want to use a custom qr, for example when your snapshots are distributed with MPI,
                    or maybe you have a fancy qr function that you can use.

        """

        self.__weighting_vector = weighting_vector
        self.__qrPicked = np.linalg.qr if qrFnc == None else qrFnc

    def __call__(self, my_array: np.ndarray):
        assert my_array.shape[0] == self.__weighting_vector.size, "Weighting vector does not match basis size"
        tmp = scipy.sparse.diags(np.sqrt(self.__weighting_vector)) @ my_array
        my_array,_ = self.__qrPicked(tmp,mode='reduced')
        my_array = scipy.sparse.diags(np.sqrt(1./self.__weighting_vector)) @ my_array
        return my_array

