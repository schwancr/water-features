import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances, get_square_displacements
from .utils import get_neighbors
from . import FeatureTrajectory
import copy

class Vectors(BaseTransformer):
    """
    Featurize the structure of water by describing the relative
    positions of the water's orientation
    
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
        hydrogens = np.array([[a.index for a in traj.top.atom(i).residue.atoms if a.element.symbol == 'H'] for i in oxygens])

        oxygen_pos = traj.xyz[:, oxygens, :]
        mean_hydrogen_pos = 0.5 * (traj.xyz[:, hydrogens[:, 0], :] + traj.xyz[:, hydrogens[:, 1], :])
        vectors = mean_hydrogen_pos - oxygen_pos
        # this is an array with a vector for each water corresponding to the 
        # orientation of the O relative to the H's
        # shape : [n_frames, n_waters, 3]

        # convert to unit vectors
        vectors = vectors / np.sqrt(np.square(vectors).sum(2, keepdims=True))

        # vec .dot. e_x = |vec| |e_x| cos(angle)
        #        vec[0] = cos(angle)
        angle_to_x_axis = np.arctan(vectors[:, :, 1] / vectors[:, :, 0]) + np.pi * (vectors[:, :, 0] < 0)
        angle_to_x_axis[np.where(vectors[:, :, 0] == 0)] = 0.0  # then we divided by zero, but it means we're
                                                                # already on the x-axis
        cos_angle = np.cos(angle_to_x_axis)
        sin_angle = np.sin(angle_to_x_axis)

        # rotation about the z-axis to get everything in the ZX plane
        Rzs = np.zeros((traj.n_frames, len(oxygens), 3, 3))
        Rzs[:, :, 2, 2] = 1
        Rzs[:, :, 0, 0] = cos_angle
        Rzs[:, :, 1, 1] = cos_angle
        Rzs[:, :, 0, 1] = sin_angle
        Rzs[:, :, 1, 0] = -1 * sin_angle
        # This is the transpose of wikipedia b/c we need to rotate clockwise

        # This is the angle to the z-axis necessary AFTER the above rotation
        angle_to_z_axis = np.arccos(vectors[:, :, 2]) 
        cos_angle = np.cos(angle_to_z_axis)
        sin_angle = np.sin(angle_to_z_axis)

        # rotation about the ziaxis to get everything in the ZY plane
        # this then puts the vectors aligned to the Z-axis if we do Rys.dot(Rxs.dot(vectors))
        Rys = np.zeros((traj.n_frames, len(oxygens), 3, 3))
        Rys[:, :, 1, 1] = 1
        Rys[:, :, 0, 0] = cos_angle
        Rys[:, :, 2, 2] = cos_angle
        Rys[:, :, 0, 2] = -1 * sin_angle
        Rys[:, :, 2, 0] = sin_angle

        vectors = vectors.reshape((traj.n_frames, 1, len(oxygens), 3))
        vectors = np.hstack([vectors] * len(oxygens))

        # this is doing matrix multiplication but broadcasting
        # the time/water dimensions
        # I don't fully understand why this works...
        rot_mats = np.einsum('...cd,...de', Rys, Rzs)
        rotated_vectors = np.einsum('...de,...ce', rot_mats, vectors)

        ##
        #i = np.random.randint(len(oxygens))
        #print rotated_vectors[:, i, i]
        ## ^^^ for all i, these are supposed to be [0, 0, 1]
        ##

        #######
        ## These tests worked when I wrote it up.
        #rot_mats_ref = np.array([[Rzs[t, nw].dot(Rys[t, nw]) 
        #                           for nw in xrange(len(oxygens))] 
        #                               for t in xrange(traj.n_frames)])
        #
        #rotated_vectors_ref = np.zeros((traj.n_frames, len(oxygens), len(oxygens), 3))
        #for t in xrange(traj.n_frames):
        #    for nw in xrange(len(oxygens)):
        #        for k in xrange(len(oxygens)):
        #            rotated_vectors_ref[t, nw, k] = rot_mats[t, nw].dot(vectors[t, nw, k])
        #
        #print np.abs(rot_mats_ref - rot_mats).max()
        #print np.abs(rotated_vectors_ref - rotated_vectors).max()
        ####################

        OOdisplacements = get_square_displacements(traj, oxygens)
        OOdistances = np.sqrt(np.square(OOdisplacements).sum(3))

        rotated_OOdisplacements = np.einsum('...de,...e', rot_mats, vectors)

        neighbors = get_neighbors(OOdistances, cutoff=self.cutoff)

        frames = np.arange(traj.n_frames).reshape((-1, 1, 1))
        waters = np.arange(len(oxygens)).reshape((1, -1, 1))

        OO_sorted_ind = np.argsort(OOdistances, axis=2)[:, :, 1 : self.n_waters + 1]

        # alright this does what I need it to do. BUUUUUUUT I still need to 
        # rotate the vectors first
        Xnew = []
        Xnew.append(OOdistances[frames, waters, OO_sorted_ind])
        Xnew.extend([rotated_OOdisplacements[frames, waters, OO_sorted_ind, i] for i in xrange(3)])
        Xnew.extend([rotated_vectors[frames, waters, OO_sorted_ind, i] for i in xrange(3)])
        Xnew = np.dstack(Xnew)

        return FeatureTrajectory(Xnew, neighbors)
