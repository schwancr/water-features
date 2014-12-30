
import multiprocessing as mp
from sklearn.base import BaseEstimator, TransformerMixin
from .featurs import FeatureTrajectory
import numpy as np


def help_me_transform(args):
    transformer = args[0]
    traj = args[1]
    return transformer.transform(traj)
    

class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base featurizer for turning simulations into vectors.

    """
    def transform(self, traj):
        pass


    def parallel_transform(self, traj, n_procs=1):
        """
        Use multiprocessing to transform in parallel
        """
        
        pool = mp.Pool(n_procs)

        try:
            result = pool.map_async(help_me_transform, 
                                    ((self, traj[i]) for i in xrange(traj.n_frames)))
            results = result.get()

        except KeyboardInterrupt:
            raise 

        except Exception as err:
            print err.message

        feats = results[0]
        for i in xrange(1, len(results)):
            feats += results[i]

        return feats
