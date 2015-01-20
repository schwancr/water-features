import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances
from .utils import get_neighbors
from . import FeatureTrajectory
import copy

class OO(BaseTransformer):
    """
    Compute the OO distances and sort them for each water molecule
    
    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters. If None, 
        then all waters are included.
    """
    def __init__(self, n_waters=None, cutoff=0.45):
        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)

        self.cutoff = float(cutoff)


    def fit(self, trajs):
        return self

    
    def transform(self, trajs):
        result = []
        for traj in trajs:
            result.append(self._transform_one(traj))
        return result


    def _transform_one(self, traj):
        """
        Transform a trajectory into the OO features

        Parameters
        ----------
        traj : mdtraj.Trajectory

        Returns
        -------
        Xnew : np.ndarray
            sorted distances for each water molecule
        distances : np.ndarray
            distances between each water molecule
        """
        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])

        distances = get_square_distances(traj, oxygens)
        neighbors = get_neighbors(distances, cutoff=self.cutoff)

        Xnew = copy.copy(distances)
        Xnew.sort(axis=2)

        if self.n_waters is None:
            Xnew = Xnew[:, :, 1:]
        else:
            Xnew = Xnew[:, :, 1:(self.n_waters + 1)]

        return FeatureTrajectory(Xnew, neighbors)
