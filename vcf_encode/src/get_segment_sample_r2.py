import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop_map',
                        type=str,
                        required=False,
                        help='Path to the pop super pop map file.')
    parser.add_argument('--pop_def',
                        type=str,
                        required=False,
                        help='Path to the pop def for samples')
    parser.add_argument('--gt',
                        type=str,
                        required=True,
                        help='Genotype distances')
    parser.add_argument('--emb',
                        type=str,
                        required=True,
                        help='Embedding distances')
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help='Output file')
    return parser.parse_args()

def get_pop_map(file):
    pop_map = {}
    with open(file, 'r') as f:
        headder = f.readline()
        for line in f:
            parts = line.strip().split()
            pop_map[parts[0]] = parts[1]

    return pop_map

def get_sample_pop_map(file, pop_map):
    sample_pop_map = {}
    with open(file, 'r') as f:
        headder = f.readline()
        for line in f:
            parts = line.strip().split()
            sample_pop_map[parts[0]] = (parts[1], pop_map[parts[1]])

    return sample_pop_map

def get_segment_data(file, sample_pop_map=None):
    dists = {}
    with open(file, 'r') as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in dists:
                dists[items[0]] = []
            if pop_map is not None and sample_pop_map is not None:
                if sample_pop_map[items[0]][1] == sample_pop_map[items[1]][1]:
                    dists[items[0]].append(float(items[2]))
            else:
                dists[items[0]].append(float(items[2]))
    return dists

def main():
    args = get_args()

    gt_dists = get_segment_data(args.gt)
    emb_dists = get_segment_data(args.emb)

    pop_map = None
    sample_pop_map = None
    if args.pop_map is not None: and args.pop_def is not None:
        pop_map = get_pop_map(args.pop_map)
        sample_pop_map = get_sample_pop_map(args.pop_def, pop_map)

    names = list(gt_dists.keys())

    with open(args.out, 'w') as f:
        f.write('sample\tr2\n')

        for sample in names:
            X_np = np.array(gt_dists[sample]).reshape(-1, 1)
            Y_np = np.array(emb_dists[sample])

            # Fit linear model
            model = LinearRegression().fit(X_np, Y_np)
            Y_pred = model.predict(X_np)
            r2 = r2_score(Y_np, Y_pred)
            f.write(f'{sample}\t{r2:.4f}\n')

if __name__ == '__main__':
    main()
