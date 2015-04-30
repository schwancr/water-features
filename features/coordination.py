import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances
from .utils import get_neighbors
from . import FeatureTrajectory
import copy

class Coordination(BaseTransformer):
    r"""
    Use the effective coordination number to compute inner
    products.
    
        .. math:: \frac{1}{\exp(\kappa (r - r_c)) + 1}

    Parameters
    ----------
    kappa : float, optional
        kappa used in the Fermi function.
    max_rc : float, optional
        The maximum :math:`r_c` to use when computing the 
        integral for the inner product. If None, then we'll
        use half the smallest box length.
    n_points : int, optional
        Number of points to evaluate the effective coordination
        number
    """
    def __init__(self, kappa=100, max_rc=1.0, n_points=100, cutoff=0.45):
        self.max_rc = float(max_rc)
        self.n_points = int(n_points)
        self.kappa = float(kappa)

        self.cutoff = float(cutoff)

        super(self.__class__, self).__init__(zscore=False)


    def _transform_one(self, traj):
        """
        Transform a trajectory into the feature space

        Parameters
        ----------
        traj : mdtraj.Trajectory

        Returns
        -------
        features : FeatureTrajectory
        """
        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])

        distances = get_square_distances(traj, oxygens)
        neighbors = get_neighbors(distances, cutoff=self.cutoff)

        Xnew = copy.copy(distances)
        Xnew.sort(axis=2)
        Xnew = Xnew[:, :, 1:]

        self.rcs_ = np.linspace(0.0, self.max_rc, self.n_points)

        Xnew = np.dstack([np.power(1 + np.exp(self.kappa * (Xnew - rc)), -1).sum(axis=2) for rc in self.rcs_])

        return FeatureTrajectory(Xnew, neighbors)
