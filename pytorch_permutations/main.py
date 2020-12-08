import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Number of possible permutations')
parser.add_argument('--classes', default=30, type=int, help='Number of permutation to select')
parser.add_argument('--selection', default='max', type=str,
                    help='Sample selected per iteration based on hamming distance: '
                         '[max] highest; [mean] average')
args = parser.parse_args()

if __name__ == "__main__":
    outname = 'permutation/permutations_hamming_%s_%d' % (args.selection, args.classes)

    # Create all possible permutations
    P_hat = np.array(list(itertools.permutations(list(range(9)), 9)))
    n = P_hat.shape[0]

    for i in trange(args.classes):
        if i == 0:
            # Take one random permutation to start
            j = np.random.randint(n)
            P = np.array((P_hat[j])).reshape([1, -1])
        else:
            # Concatenate P with the max distance permutation
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        # Remove from all permutation the selected permutation
        P_hat = np.delete(P_hat, j, axis=0)
        # Compute distance between all of the permutation remaining
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        if args.selection == 'max':
            # Select max distance
            j = D.argmax()
        else:
            # Average Distance
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]

    np.save(outname, P)
    print('file created -->' + outname)
