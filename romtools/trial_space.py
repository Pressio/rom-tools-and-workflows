"""
#Trial space overview

A trial space is foundational to reduced-order models.
In a ROM, we restrict a high-dimensional state to live within a low-dimensional trial space.
Mathematically, for a "FOM" vector $\\mathbf{u} \\in \\mathbb{R}^N$, we represent this as
$$\\mathbf{u} \\approx \\tilde{\\mathbf{u}} \\in \\mathcal{V}$$
where $\\mathcal{V}$ with
$\\text{dim}(\\mathcal{V}) = K \\le N$
is the trial space. Formally, we can describe this low-dimensional representation with a basis and an offine offset,
$$\\tilde{\\mathbf{u}}  = \\boldsymbol \\Phi \\hat{\\mathbf{u}} + \\mathbf{u}_{\\mathrm{shift}}$$
where $\\boldsymbol \\Phi \\in \\mathbb{R}^{N \times K}$ is the basis matrix,
$\\hat{\\mathbf{u}} \\in \\mathbb{R}^{K}$ are the reduced, or generalized coordinates,
$\\mathbf{u}_{\\mathrm{shift}} \\in \\mathbb{R}^N$ is the shift vector (or affine offset), and, by definition,
$\\mathcal{V} \\equiv \\mathrm{range}(\\boldsymbol \\Phi) + \\mathbf{u}_{\\mathrm{shift}}$.

The trial_space class encapsulates the information of an affine trial space, $\\mathcal{V}$,
by virtue of providing access to a basis matrix, a shift vector, and the dimensionality of the trial space.
"""
import abc
import numpy as np
from romtools.snapshot_data import AbstractSnapshotData
from romtools.trial_space_utils.truncater import AbstractTruncater, NoOpTruncater
from romtools.trial_space_utils.shifter import AbstractShifter, NoOpShifter
from romtools.trial_space_utils.scaler import AbstractScaler, NoOpScaler
from romtools.trial_space_utils.splitter import AbstractSplitter, NoOpSplitter
from romtools.trial_space_utils.orthogonalizer import AbstractOrthogonalizer, NoOpOrthogonalizer

class AbstractTrialSpace(abc.ABC):
    """Abstract implementation"""
    @abc.abstractmethod
    def __init__(snapshots: AbstractSnapshotData):
        pass

    @abc.abstractmethod
    def getDimension(self):
        """Returns the dimension of the trial space"""
        pass

    @abc.abstractmethod
    def getShiftVector(self):
        """Returns the shift vector"""
        pass

    @abc.abstractmethod
    def getBasis(self):
        """Returns the basis"""
        pass



class DictionaryTrialSpace(AbstractTrialSpace):
    """
    ##Reduced basis trial space (no truncation).

    Given a snapshot matrix $\\mathbf{S}$, we set the basis to be

    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathrm{split}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}}))$$

    where the orthogonalization, splitting, and shifts are defined by their respective classes
    """
    def __init__(self,snapshot_data,shifter,splitter,orthogonalizer):
        # inputs:
        # fom_data: snapshot_data object, contains lists of full model solution data, methods to read it
        #           and other metadata such as variable set type
        # shifter: class that shifts the basis
        # splitter: class that splits basis
        # orthogonalizer: class that orthogonalizes basis

        # compute basis
        snapshots = snapshot_data.getSnapshotsAsArray()
        shifted_snapshots,self.__shift_vector = shifter(snapshots)
        self.__basis = splitter(shifted_snapshots)
        self.__basis = orthogonalizer(self.__basis)
        self.__dimension = self.__basis.shape[1]

    def getDimension(self):
      """Returns dimension of trial space"""
      return self.__dimension

    def getShiftVector(self):
      """Returns the shift vector"""
      return self.__shift_vector

    def getBasis(self):
       """Returns the basis"""
       return self.__basis


class TrialSpaceFromPOD(AbstractTrialSpace):
    """
    ##POD trial space (constructed via SVD).

    Given a snapshot matrix $\\mathbf{S}$, we set the basis to be

    $$\\boldsymbol U = \\mathrm{SVD}(\\mathrm{split}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}})))$$
    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathrm{truncate}( \\boldsymbol U ))$$

    where $\\boldsymbol U$ are the left singular vectors and the orthogonalization,
    truncation, splitting, and shifts are defined by their respective classes.

    For truncation, we enable truncation based on a fixed dimension or the decay
    of singular values; please refer to the documentation for the truncater.
    """

    def __init__(self,
                 snapshots:      AbstractSnapshotData,
                 truncater:      AbstractTruncater      = NoOpTruncater(),
                 shifter:        AbstractShifter        = NoOpShifter(),
                 splitter:       AbstractSplitter       = NoOpSplitter(),
                 orthogonalizer: AbstractOrthogonalizer = NoOpOrthogonalizer(),
                 svdFnc = None):
        """
        Constructor

        Args:

            snapshots (AbstractSnapshotData): snapshot data

            truncater (AbstractTruncater): object for truncating the basis.

            shifter (AbstractShifter): object for shifting the basis

            splitter (AbstractSplitter): object for splitting the basis

            orthogonalizer (AbstractOrthogonalizer): object for orthogonalize the basis

            svdFnc: a callable to use for computing the SVD on the snapshots data.
                    IMPORTANT: must conform to the API of [np.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html#numpy-linalg-svd).
                    If `None`, internally we use `np.linalg.svd`.
                    Note: this is useful when you want to use a custom svd, for example when your snapshots are distributed with MPI,
                    or maybe you have a fancy svd function that you can use.

        """

        snapshots = snapshots.getSnapshotsAsArray()
        shifted_snapshots, self.__shift_vector = shifter(snapshots)
        shifted_split_snapshots = splitter(shifted_snapshots)

        svdPicked = np.linalg.svd if svdFnc == None else svdFnc
        lsv, svals, _ = svdPicked(shifted_split_snapshots, full_matrices=False, \
                                  compute_uv=True, hermitian=False)

        self.__basis = truncater(lsv, svals)
        self.__basis = orthogonalizer(self.__basis)
        self.__dimension = self.__basis.shape[1]

    def getDimension(self):
        """Returns dimension of trial space"""
        return self.__dimension

    def getShiftVector(self):
        """Returns the shift vector"""
        return self.__shift_vector

    def getBasis(self):
        """Returns the basis"""
        return self.__basis


class TrialSpaceFromScaledPOD(AbstractTrialSpace):
    """
    ##POD trial space (constructed via scaled SVD).

    Given a snapshot matrix $\\mathbf{S}$, we set the basis to be

    $$\\boldsymbol U = \\mathrm{SVD}(\\mathrm{split}(\\mathrm{prescale}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}})))$$
    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathrm{postscale}(\\mathrm{truncate}( \\boldsymbol U )))$$

    where $\\boldsymbol U$ are the left singular vectors and the orthogonalization,
    truncation, splitting, and shifts are defined by their respective classes.

    For truncation, we enable truncation based on a fixed dimension or the decay of singular values;
    please refer to the documentation for the truncater.
    """

    def __init__(self, snapshot_data: AbstractSnapshotData,
                 truncater: AbstractTruncater,
                 shifter: AbstractShifter,
                 scaler: AbstractScaler,
                 splitter: AbstractSplitter,
                 orthogonalizer: AbstractOrthogonalizer):
        # inputs:
        # fom_data: snapshot_data object, contains lists of full model solution data, methods to read it
        #           and other metadata such as variable set type
        # truncater: class that truncates the basis
        # shifter: class that shifts the basis
        # scaler: class the scales

        # compute basis
        snapshots = snapshot_data.getSnapshotsAsArray()
        shifted_snapshots,self.__shift_vector = shifter(snapshots)
        scaled_shifted_snapshots = scaler.preScaling(shifted_snapshots)
        scaled_shifted_and_split_snapshots = splitter(scaled_shifted_snapshots)
        lsv,svals,_ = np.linalg.svd(scaled_shifted_and_split_snapshots,full_matrices=False)
        self.__basis = truncater(lsv,svals)
        self.__basis = scaler.postScaling(self.__basis)
        self.__basis = orthogonalizer(self.__basis)
        self.__dimension = self.__basis.shape[1]

    def getDimension(self):
        """Returns dimension of trial space"""
        return self.__dimension

    def getShiftVector(self):
        """Returns the shift vector"""
        return self.__shift_vector

    def getBasis(self):
        """Returns the basis"""
        return self.__basis
