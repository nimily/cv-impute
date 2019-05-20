from argparse import ArgumentParser

import numpy as np

import matplotlib.pyplot as plt

from utils import plot_data
from utils import moments_to_confidence_band


def load_data(data_file):
    stats = {}

    with open(data_file, 'r') as f:
        # skip the header
        f.readline()

        for line in f:
            _, name, n_sample, error = line.split(',')

            n_sample = int(n_sample)
            error = float(error)

            if name not in stats:
                stats[name] = {}

            if n_sample not in stats[name]:
                stats[name][n_sample] = [0, 0, 0]

            stats[name][n_sample][0] += 1
            stats[name][n_sample][1] += error
            stats[name][n_sample][2] += error ** 2

    data = {}

    for name, stat in stats.items():
        xs = sorted(stat.keys())
        ms = np.array([stat[x] for x in xs])

        ys, ss = moments_to_confidence_band(ms[:, 0], ms[:, 1], ms[:, 2])
        data[name] = (xs, ys, ss)

    return data


def __main__():
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--plot', type=str, default=None)

    args = parser.parse_args()
    if args.name is not None:
        data_file = f'outputs/{args.name}.csv'
        plot_file = f'plots/{args.name}.pdf'
    else:
        data_file = args.data
        plot_file = args.plot

    data = load_data(data_file)
    plot_data(data, plt)

    if plot_file is None:
        plt.show()
    else:
        plt.savefig(plot_file)


if __name__ == '__main__':
    __main__()
