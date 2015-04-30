import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances, get_square_displacements
from .utils import get_neighbors
from . import FeatureTrajectory
import scipy.special
import copy

class SphericalHarmonics(BaseTransformer):
    """
    Featurize the structure of water by describing the relative
    positions of the water's orientation using spherical harmonics
    
    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters. If None, 
        then all waters are included.
    """
    def __init__(self, n_waters=None, cutoff=0.45, harmonics=10,
                 zscore=False):
        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)

        self.cutoff = float(cutoff)

        if isinstance(harmonics, list):
            self.harmonics = harmonics
        else:
            self.harmonics = range(int(harmonics))

        super(self.__class__, self).__init__(zscore=zscore)


    def _transform_one(self, traj):
        """
        Transform a trajectory into the spherical harmonics
        featurization.

        Parameters
        ----------
        traj : mdtraj.Trajectory

        Returns
        -------
        Xnew : features.FeatureTrajectory
            Feature trajectory containing the featurization
            and neighborlists.
        """
        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])
        hydrogens = np.array([[a.index for a in traj.top.atom(i).residue.atoms if a.element.symbol == 'H'] for i in oxygens])

        oxygen_pos = traj.xyz[:, oxygens, :]
        mean_hydrogen_pos = 0.5 * (traj.xyz[:, hydrogens[:, 0], :] + traj.xyz[:, hydrogens[:, 1], :])
        HO_vectors = mean_hydrogen_pos - oxygen_pos
        # this is an array with a vector for each water corresponding to the 
        # orientation of the O relative to the H's
        HH_vectors = traj.xyz[:, hydrogens[:, 0], :] - traj.xyz[:, hydrogens[:, 1], :]
        # this one is H relative to the other H. Obviously there is a degeneracy, but 
        # we're going to sum up over all quadrants, so it should end up being equal. 
        # plus we're just trying to get the water in the XZ plane
        # Both have the same shape : [n_frames, n_waters, 3]

        # convert to unit vectors
        HO_vectors = HO_vectors / np.sqrt(np.square(HO_vectors).sum(2, keepdims=True))
        #HH_vectors = HH_vectors / np.sqrt(np.square(HH_vectors).sum(2, keepdims=True))

        # I need to make the HH perpindicular to HO. Since I'm doing it this way, the 
        # water may be a bit crooked, but the HH -> O always points up
        HH_vectors = HH_vectors - (HH_vectors * HO_vectors).sum(2, keepdims=True) * HO_vectors
        # this is just the gram-schmidt process for 2 vectors
        # I have to normalize it now
        HH_vectors = HH_vectors / np.sqrt(np.square(HH_vectors).sum(2, keepdims=True))
        
        HO_vectors = HO_vectors.reshape((traj.n_frames, len(oxygens), 1, 3))
        HH_vectors = HH_vectors.reshape((traj.n_frames, len(oxygens), 1, 3))
        # I need to add an axis so that these vectors are broadcast to every water in 
        # the row of the displacement matrix

        # Ok, so HO_vectors is my z-axis, and HH_vectors is my x-axis (but degenerate)
        # Now, I need to compute the displacement vectors' angles relative to these axes
        # And I will use these angles to plug into the spherical harmonics
        # I don't ever need to rotate anything ...
        OOdisplacements = get_square_displacements(traj, oxygens)
        OOdistances = np.sqrt(np.square(OOdisplacements).sum(3))
        OOdisplacements = OOdisplacements / np.sqrt(np.square(OOdisplacements).sum(3, keepdims=True))
        # normalize to make the angle calculation just a dot product

        # The polar angle is just the angle to the z-axis:
        # I should find out if np.einsum is faster than this operation
        # actually, the sph_harm stuff is the slow step, so I'd rather 
        # leave this in a more readible format
        polar_comp = (HO_vectors * OOdisplacements).sum(3, keepdims=True) # we'll squeeze this last row later

        # Now, the azimuthal angle is a bit harder, because it's the angle of the displacement
        # projcted onto the XY plane

        # first, subtract out the z-component and normalize again
        XY_displacements = OOdisplacements - polar_comp * HO_vectors
        XY_displacements = XY_displacements / np.sqrt(np.square(XY_displacements).sum(axis=3, keepdims=True))

        polar_angle = np.arccos(polar_comp[:, :, :, 0])  
        # we kept the dims before in order to broadcast correctly.

        # now, take the dot product with the HH_vectors:
        #azim_comp = np.abs((HH_vectors * XY_displacements).sum(3))
        # ^^^ so I'm already screwed b/c I'm not figuring out whether
        # the angle is in [0, pi] or [pi, 2pi], so I might as well just
        # only describe the angle in the first quadrant. 
        # I have no idea if this is the right thing to do or not, but 
        # it might be. How does limiting my points to the same quadrant
        # affect the spherical harmonics stuff?
        # Actually, I think this is wrong, since most of the spherical
        # harmonics lack this symmetry. I don't know how to account for
        # all of the symmetries that water has. It seems like these
        # spherical harmonics just simply don't work.
        azim_comp = (HH_vectors * XY_displacements).sum(3)
        azim_angle = np.arccos(azim_comp)

        # ok, so now [azim,polar]_angle is shaped [n_frames, n_waters, n_waters]
        # and [azim,polar]_angle[t, i, j] is the [azim,polar] angle of the vector
        # between water i and water j at time t, in water i's molecular frame
        neighbors = get_neighbors(OOdistances, cutoff=self.cutoff)

        frames = np.arange(traj.n_frames).reshape((-1, 1, 1))
        waters = np.arange(len(oxygens)).reshape((1, -1, 1))

        OO_sorted_ind = np.argsort(OOdistances, axis=2)[:, :, 1 : self.n_waters + 1]

        X_azim = azim_angle[frames, waters, OO_sorted_ind]
        X_polar = polar_angle[frames, waters, OO_sorted_ind]

        Xnew = [OOdistances[frames, waters, OO_sorted_ind]]
        for l in self.harmonics:
            for m in xrange(-l, l + 1):
                print "Working on l=%d, m=%d" % (l, m)
                qlm = scipy.special.sph_harm(m, l, X_azim, X_polar).sum(axis=2, keepdims=True) / 4
                #Xnew.append(np.sqrt(temp.sum(axis=2, keepdims=True)))
                Xnew.append(qlm)

        Xnew = np.dstack(Xnew)
        return FeatureTrajectory(Xnew, neighbors)
