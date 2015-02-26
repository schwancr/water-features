import mdtraj as md
import numpy as np

def get_square_displacements(traj, aind=None):
    """
    The atom displacements for a subset of the atoms in a
    trajectory. Return it as a 3-d matrix: [n_atoms, n_atoms, 3]

    Parameters
    ----------
    traj : mdtraj.Trajectory
        trajectory object
    aind : np.ndarray, optional
        distances between the atoms in aind

    Returns
    -------
    displacements : np.ndarray, 
                    shape = [traj.n_frames, aind.shape[0], aind.shape[0], 3]
        displacements between thea toms in aind
    """
    if aind is None:
        aind = np.arange(traj.n_atoms)

    pairs_ind = np.array([(i, j) for i in xrange(len(aind)) for j in xrange(i + 1, len(aind))])
    pairs = np.array([(aind[i], aind[j]) for i, j in pairs_ind])

    displacements = md.compute_distances(traj, pairs)
    displacements = np.array([md.geometry.squareform(displacements[:, :, i], pairs_ind) for i in xrange(3)])
    displacements = displacements.transpose((1, 2, 3, 0))
    # displacements is now shaped like we want it
    
    return displacements


def get_square_distances(traj, aind=None):
    """
    The atom distances in for a subset of the atoms in
    a trajectory and return it as a square distance matrix

    Parameters
    ----------
    traj : mdtraj.Trajectory 
        trajectory object
    aind : np.ndarray, optional
        atom indices to compute distances between

    Returns
    -------
    distances : np.ndarray, shape = [traj.n_frames, aind.shape[0], aind.shape[0]]
        distances between the atoms in aind
    """

    if aind is None:
        aind = np.arange(traj.n_atoms)

    pairs_ind = np.array([(i, j) for i in xrange(len(aind)) for j in xrange(i + 1, len(aind))])
    pairs = np.array([(aind[i], aind[j]) for i, j in pairs_ind])

    distances = md.compute_distances(traj, pairs)
    distances = md.geometry.squareform(distances, pairs_ind)

    return distances


def get_neighbors(distances, cutoff=0.45):
    """
    Get the neighbor lists for each molecule in an array of distances

    Parameters
    ----------
    distances : np.ndarray, shape = [n_frames, n_molecules, n_molecules]
        Distances between waters in each frame. distances[t, i, j] is the 
        distance between molecule i and molecule j in frame t
    cutoff : float, optional
        Cutoff to construct the neighbor lists (nm).

    Returns
    -------
    neighbors : list, shape = [n_frames, n_molecules, variable]
        List of lists where neighbors[t][i][k] is the kth closest
        molecule to molecule i at frame t
    """

    is_close = (distances <= cutoff)

    # pretty sure there's a clever way to do this, but I 
    # haven't found it yet...
    neighbors = []
    for frame in is_close:
        molecules = []
        for mol in frame:
            molecules.append(np.where(mol)[0])
        neighbors.append(molecules)

    return neighbors
