import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import gridspec
from scipy.stats import gaussian_kde

def get_args():
    parser = argparse.ArgumentParser(description='Get per sample desnity and r^2')
    parser.add_argument('--gt',
                        type=str,
                        required=True,
                        help='Input file containing pair-wise gt distancces.')
    parser.add_argument('--emb',
                        type=str,
                        required=True,
                        help='Input file containing pair-wise emb distances.')
    parser.add_argument('-d', '--density',
                        type=str,
                        required=True,
                        help='Input file containing density.')
    parser.add_argument('-n', '--names',
                        type=str,
                        required=True,
                        help='Input file containing sample names.')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='Output file to save the correlation plot.')
    parser.add_argument('--height',
                        type=int,
                        default=5,
                        help='Height of the plot.')
    parser.add_argument('--width',
                        type=int,
                        default=5,
                        help='Width of the plot.')
    parser.add_argument('--segment',
                        type=str,
                        help='Segment Number.')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose mode.')
    return parser.parse_args()

def get_sample_names(file):
    names = []
    with open(file, 'r') as f:
        for line in f:
            items = line.strip().split()
            names.append(items[0] + '_0')
            names.append(items[0] + '_1')
    return names

def get_sample_densities(file):
    desnities = []
    with open(file, 'r') as f:
        for line in f:
            items = line.strip().split()
            desnities.append(float(items[0]))
    return desnities

def get_segment_data(file):
    #print(f'Loading {file}...')
    start_time = time.time()
    dists = {}
    with open(file, 'r') as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in dists:
                dists[items[0]] = []
            dists[items[0]].append(float(items[2]))
    #print(f'Loaded {len(dists)} distances in {time.time() - start_time:.2f}s')
    return dists

def main():
    args = get_args()

    names = get_sample_names(args.names)
    density = get_sample_densities(args.density)

    gt_dists = get_segment_data(args.gt)
    emb_dists = get_segment_data(args.emb)

    r2s = []
    densities = []
    for i, sample in enumerate(names):
        X_np = np.array(gt_dists[sample]).reshape(-1, 1)
        Y_np = np.array(emb_dists[sample])

        # Fit linear model
        model = LinearRegression().fit(X_np, Y_np)
        Y_pred = model.predict(X_np)
        r2 = r2_score(Y_np, Y_pred)
        if args.verbose:
            print(f'{r2:.4f} {density[i]}')
        r2s.append(r2)
        densities.append(density[i])

    if args.output:
        fig = plt.figure(figsize=(args.width, args.height))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[1, 1, 1, 0.3],  # last column (y KDE) is narrow
                               height_ratios=[0.3, 1, 1, 1]  # top row (x KDE) is short
        )

        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_xkde = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_ykde = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

        # Main scatter
        ax_main.scatter(densities, r2s, alpha=0.1, label='Data')
        ax_main.set_xlabel('Density')
        ax_main.set_ylabel('$R^2$')
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)

        # Compute KDEs
        density_kde = gaussian_kde(densities)
        r2_kde = gaussian_kde(r2s)

        x_vals = np.linspace(min(densities), max(densities), 200)
        y_vals = np.linspace(min(r2s), max(r2s), 200)

        # Top KDE (Density)
        ax_xkde.fill_between(x_vals, density_kde(x_vals), color='orange', alpha=0.3)
        ax_xkde.plot(x_vals, density_kde(x_vals), color='orange')
        ax_xkde.axis('off')
        ax_xkde.set_title(f'Segment: {args.segment}', fontsize=10)

        # Right KDE (R²)
        ax_ykde.fill_betweenx(y_vals, r2_kde(y_vals), color='orange', alpha=0.3)
        ax_ykde.plot(r2_kde(y_vals), y_vals, color='orange')
        ax_ykde.axis('off')

        plt.tight_layout()
        plt.savefig(args.output)

if __name__ == '__main__':
    main()
