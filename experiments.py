from collections import namedtuple

import numpy as np
import numpy.random as npr

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

Config = namedtuple('Config', 'n_row, n_col, rank, sd')


def init_problem(shape, train=None, test=None):
    imputer = SoftImpute(shape, tuned_svt())

    if train is None:
        train = Dataset(LinOp(shape), [])

    if test is None:
        test = Dataset(LinOp(shape), [])

    return Problem(imputer, train, test)


def run_single(seed, config, sizes, alphas, n_fold, fout=None):
    npr.seed(seed)

    shape = config.n_row, config.n_col

    bl = npr.randn(config.n_row, config.rank)
    br = npr.randn(config.n_col, config.rank)
    b = bl @ br.T

    ground_truth = Dataset(LinOp(shape), [])
    ground_truth.extend(
        xs=np.ndindex(shape + (1,)),
        ys=b.flat
    )

    max_size = max(sizes)
    rs = npr.randint(config.n_row, size=max_size)
    cs = npr.randint(config.n_col, size=max_size)
    vs = np.ones(max_size, dtype=np.int)
    xs = np.hstack([
        rs[:, np.newaxis],
        cs[:, np.newaxis],
        vs[:, np.newaxis]
    ]).tolist()
    ys = b[rs, cs] + npr.randn(max_size) * config.sd

    names = ['cv', 'oracle', 'regular']
    stats = {name: [] for name in names}

    for i, size in enumerate(sizes):
        print(f'round {i} with size {size} has started...')

        xss = xs[:size]
        yss = ys[:size]

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

        bh, *_ = oracle_impute(prob)
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
                fout.write(f'{seed},{name},{size},{errors[-1]}')

    return {name: np.array(perf) for name, perf in stats.items()}


def __main__(fout):
    config = Config(
        n_row=100,
        n_col=100,
        rank=3,
        sd=1.0
    )
    n_fold = 10
    step = 100
    n_sample = 1000

    sizes = list([i + step for i in range(0, n_sample, step)])

    print('generating the sequence of alphas...')
    alphas = [compute_op_norm_thresh(config, size, level=0.9, repetition=10) for size in sizes]

    # aggregating the stats
    aggs = {}

    for seed in range(10):
        stats = run_single(seed, config, sizes, alphas, n_fold, fout)

        for name, perf in stats.items():
            if name not in aggs:
                aggs[name] = [np.zeros_like(sizes, dtype=np.float64) for _ in range(3)]

            aggs[name][0] += 1
            aggs[name][1] += perf
            aggs[name][2] += perf ** 2


if __name__ == '__main__':
    out = 'stats.csv'
    with open(out, 'w') as fout:

        header = ','.join(['seed', 'name', 'n_sample', 'error'])
        fout.write(header)

        __main__(fout)
