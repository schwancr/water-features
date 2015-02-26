
import multiprocessing as mp
from sklearn.base import BaseEstimator, TransformerMixin
from . import FeatureTrajectory
import numpy as np


def help_me_transform(args):
    transformer = args[0]
    traj = args[1]
    return transformer.transform(traj)
    

class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base featurizer for turning simulations into vectors.

    """
    def __init__(self, zscore=True):
        self.zscore = zscore
        self.means_ = None
        self.stds_ = None


    def fit(self, trajs, y=None):
        return self


    def transform(self, trajs):
        result = []
        for traj in trajs:
            result.append(self._transform_one(traj))

        if self.zscore:
            if self.means_ is None:
                self.set_zscore(result)

            result = self.apply_zscore(result)

        return result


    def set_zscore(self, X):
        features = np.concatenate([np.concatenate(f.features) for f in X])
        
        self.means_ = features.mean(0)
        self.stds_ = features.std(0)


    def apply_zscore(self, Xs):
        Zs = []
        for X in Xs:
            Z = X
            Z.features -= self.means_.reshape((1, 1, -1))
            Z.features /= self.stds_.reshape((1, 1, -1))
            Zs.append(Z)

        return Zs
