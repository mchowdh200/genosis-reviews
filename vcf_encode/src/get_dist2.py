import argparse
import numpy as np
import gzip
from scipy.spatial.distance import pdist, squareform

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to gzipped input file')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to output file')
    parser.add_argument('-c', '--col', default=1, type=int, help='Index of first column with numeric features (default: 1)')
    return parser.parse_args()

def main():
    args = get_args()

    sample_ids = []
    data = []

    # Read gzipped input file and parse data
    with gzip.open(args.input, 'rt') as f:
        for line in f:
            tokens = line.strip().split()
            sample_ids.append(tokens[0])
            features = [float(x) for x in tokens[args.col:]]
            data.append(features)

    data = np.array(data)
    num_samples = len(sample_ids)

    # Compute pairwise distances (condensed format) efficiently
    condensed_dists = pdist(data, metric='euclidean')
    dist_matrix = squareform(condensed_dists)

    # Write pairwise distances to output (excluding self-distances)
    with open(args.output, 'w') as out:
        for i in range(num_samples):
            for j in range(num_samples):
                if i == j:
                    continue
                out.write(f"{sample_ids[i]} {sample_ids[j]} {dist_matrix[i, j]:.4f}\n")

if __name__ == '__main__':
    main()

