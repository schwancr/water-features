import numpy as np
import mdtraj as md
from .base import BaseTransformer
from .utils import get_square_distances, get_neighbors
from . import FeatureTrajectory
import copy


class OOneighbors(BaseTransformer):
    """
    Compute the OO distances and sort them for each water molecule
    
    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters. If None, 
        then all waters are included.
    """
    def __init__(self, n_waters=None, cutoff=0.45, zscore=True):
        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)

        self.cutoff = float(cutoff)
        
        super(OOneighbors, self).__init__(zscore=zscore)


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
        Xnew = copy.copy(distances)
        Xnew.sort()

        neighbors = get_neighbors(distances, cutoff=self.cutoff)

        if self.n_waters is None:
            Xnew = Xnew[:, :, 1:]
        else:
            Xnew = Xnew[:, :, 1:(self.n_waters + 1)]

        sorted_waters = np.argsort(distances, axis=-1)
        # sorted_waters[t, i, k] contains the k'th closest water index to water i at time t
        # k==0 is clearly i

        ind0 = np.array([np.arange(Xnew.shape[0])] * Xnew.shape[1]).T

        Xnew0 = copy.copy(Xnew)

        for k in xrange(1, 5):
            Xnew = np.concatenate([Xnew, Xnew0[ind0, sorted_waters[:, :, k]]], axis=2)

        return FeatureTrajectory(Xnew, neighbors)
