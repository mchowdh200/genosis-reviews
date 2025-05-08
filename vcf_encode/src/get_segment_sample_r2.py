import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt',
        type=str,
        required=True,
        help='Genotype distances'
    )
    parser.add_argument(
        '--emb',
        type=str,
        required=True,
        help='Embedding distances'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output file'
    )
    return parser.parse_args()

def get_segment_data(file):
    start_time = time.time()
    dists = {}
    with open(file, 'r') as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in dists:
                dists[items[0]] = []
            dists[items[0]].append(float(items[2]))
    return dists

def main():
    args = get_args()

    gt_dists = get_segment_data(args.gt)
    emb_dists = get_segment_data(args.emb)

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
