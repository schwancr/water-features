
import numpy as np

class FeatureTrajectory(object):
    """
    Feature trajectory containing a set of moving particles.
    
    Parameters
    ----------
    features : np.ndarray, shape = [n_frames, n_molecules, n_features]
        Features for each molecule.
    neighbors : list of lists
        Neighbor lists for each water molecule in each frame
    """
    def __init__(self, features, neighbors):
        features = np.array(features)
        if features.ndim != 3:
            raise ValueError("features must have three dimensions")

        self.features = features
        self.neighbors = neighbors


    @property
    def n_frames(self):
        return self.features.shape[0]


    @property
    def n_molecules(self):
        return self.features.shape[1]


    @property
    def n_features(self):
        return self.features.shape[2]


    def __iadd__(self, other):
        self.features = np.concatenate([self.features, other.features])  # these are arrays
        self.neighbors = self.neighbors + other.neighbors  # these are lists

        return self


    def __add__(self, other):
        features = np.concatenate([self.features, other.features])
        neighbors = self.neighbors + other.neighbors

        return FeatureTrajectory(features, neighbors)
