from argparse import ArgumentParser
from collections import namedtuple

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

from impute import Dataset
from impute import SoftImpute
from impute import EntryTraceLinearOp as LinOp
from impute.svt import tuned_svt

from utils import Problem
from utils import cv_impute
from utils import oracle_impute
from utils import regular_impute
from utils import relative_error
from utils import compute_op_norm_thresh
from utils import split_and_add_to_subproblems

from utils import plot_data
from utils import moments_to_confidence_band

Config = namedtuple('Config', 'n_row, n_col, rank, sd')


def init_problem(shape, train=None, test=None):
    imputer = SoftImpute(shape, tuned_svt())

    if train is None:
        train = Dataset(LinOp(shape), [])

    if test is None:
        test = Dataset(LinOp(shape), [])

    return Problem(imputer, train, test)


def int_array_to_entries(n_col, arr):
    rs = arr % n_col
    cs = arr // n_col
    vs = np.ones_like(rs)
    return np.hstack([
        rs[:, np.newaxis],
        cs[:, np.newaxis],
        vs[:, np.newaxis]
    ])


def get_noisy_obs(b, xs, sd):
    rs = xs[:, 0]
    cs = xs[:, 1]
    return b[rs, cs] + npr.randn(xs.shape[0]) * sd


def run_single(seed, config, sizes, alphas, n_fold, fout=None):
    npr.seed(seed)

    shape = config.n_row, config.n_col

    bl = npr.randn(config.n_row, config.rank)
    br = npr.randn(config.n_col, config.rank)
    b = bl @ br.T

    entries = np.arange(b.size)
    xs = int_array_to_entries(config.n_col, entries)
    ys = get_noisy_obs(b, xs, config.sd)

    ground_truth = Dataset(LinOp(shape), [])
    ground_truth.extend(
        xs=xs.tolist(),
        ys=ys.tolist()
    )

    max_size = max(sizes)
    entries = npr.choice(b.size, (max_size,), replace=False)
    xs = int_array_to_entries(config.n_col, entries)
    ys = get_noisy_obs(b, xs, config.sd)

    names = ['cv', 'oracle', 'regular']
    stats = {name: [] for name in names}

    for i, size in enumerate(sizes):
        print(f'round {i} with size {size} has started...')

        xss = xs[:size].tolist()
        yss = ys[:size].tolist()

        # cross-validation
        probs = [init_problem(shape) for _ in range(n_fold)]
        split_and_add_to_subproblems(probs, xss, yss)

        bh, *_ = cv_impute(probs)
        error = relative_error(b, bh)
        stats['cv'].append(error)
        print(f'cv relative error = {error}')

        # oracle
        prob = init_problem(shape, test=ground_truth)
        prob.train.extend(xss, yss)

        bh, *_ = oracle_impute(prob, alphas[i] / 2)
        error = relative_error(b, bh)
        stats['oracle'].append(error)
        print(f'oracle relative error = {error}')

        # regular
        prob = init_problem(shape, test=ground_truth)
        prob.train.extend(xss, yss)

        bh, *_ = regular_impute(prob, alphas[i])
        error = relative_error(b, bh)
        stats['regular'].append(error)
        print(f'regular relative error = {error}')

        # appending to the output file (if given)
        if fout is not None:
            for name, errors in stats.items():
                print(f'{seed},{name},{size},{errors[-1]}', file=fout, flush=True)

    return {name: np.array(perf) for name, perf in stats.items()}


def run_all(seed, n_run, config, step, max_size, n_fold, fout):
    sizes = [i + step for i in range(0, max_size, step)]

    print('generating the sequence of alphas...')
    alphas = [compute_op_norm_thresh(config, size, level=0.9, repetition=10) for size in sizes]

    # aggregating the stats
    aggs = {}

    for run in range(n_run):
        stats = run_single(seed + run, config, sizes, alphas, n_fold, fout)

        for name, perf in stats.items():
            if name not in aggs:
                aggs[name] = [0] * 3

            aggs[name][0] += 1
            aggs[name][1] += perf
            aggs[name][2] += perf ** 2

    data = {}
    for name, ms in aggs.items():
        xs = sizes
        ys, ss = moments_to_confidence_band(*ms)

        data[name] = (xs, ys, ss)

    plot_data(data, plt)


def __main__():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--mnr', nargs='+', type=int, default=(100, 100, 3))
    parser.add_argument('--sd', type=float, default=1.)
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--n-fold', type=int, default=10)
    parser.add_argument('--max-size', type=int, default=2000)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()

    config = Config(
        n_row=args.mnr[0],
        n_col=args.mnr[1],
        rank=args.mnr[2],
        sd=args.sd
    )

    if args.name is not None:
        with open(f'outputs/{args.name}.csv', 'w') as fout:
            header = ','.join(['seed', 'name', 'n_sample', 'error'])
            print(header, file=fout, flush=True)

            run_all(args.seed, args.run, config, args.step, args.max_size, args.n_fold, fout)

            plt.savefig(f'plots/{args.name}.pdf')
    else:
        run_all(args.seed, args.run, config, args.step, args.max_size, args.n_fold, None)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    __main__()
