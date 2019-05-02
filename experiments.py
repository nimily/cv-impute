from typing import List, NamedTuple

import numpy as np
import numpy.random as npr

from impute import Dataset
from impute import SoftImpute
from impute import EntryTraceLinearOp as LinOp

from impute.utils import SVD


def rss(y, yh) -> float:
    return np.sum((y - yh) ** 2).item()


def compute_rel_error(b, bh):
    return rss(b, bh) / rss(b, 0)


class Problem(NamedTuple):

    imputer: SoftImpute
    train: Dataset
    test: Dataset

    def fit(self, alphas=None):
        return self.imputer.fit(self.train, alphas=alphas)

    def eval_prediction(self, yh):
        ys = self.test.ys
        return rss(ys, yh)

    def eval_estimate(self, mat):
        yh = self.test.op(mat)

        return self.eval_prediction(yh)

    def fit_and_eval(self, alphas):
        svds = self.fit(alphas)
        mats = [svd.to_matrix() for svd in svds]
        errs = np.array([self.eval_estimate(mat) for mat in mats])

        return Solution(svds, mats, errs)


class Solution(NamedTuple):
    svds: List[SVD]
    mats: List[np.ndarray]
    errs: np.ndarray


def generate_foldids(n, k, shuffle=True):
    indices = np.arange(n)
    if shuffle:
        npr.shuffle(indices)

    l = n // k
    c = n % k

    starts = [f * l + min(f, c) for f in range(k)]
    ends = [(f + 1) * l + min(f + 1, c) for f in range(k)]

    folds = [indices[s:e] for s, e in zip(starts, ends)]
    return folds


def subarray(arr, ind):
    return [arr[i] for i in ind]


def init_problem(shape):
    imputer = SoftImpute(shape)
    train = Dataset(LinOp(shape), [])
    test = Dataset(LinOp(shape), [])

    return Problem(imputer, train, test)


def cv_impute(shape, xs, ys, k=10, alphas=None, n_alpha=100, alpha_min_ratio=0.0001, foldids=None, shuffle=True):
    assert len(xs) == len(ys)

    n = len(xs)
    if not foldids:
        foldids = generate_foldids(n, k, shuffle)

    folds = [(subarray(xs, fold), subarray(ys, fold)) for fold in foldids]

    probs = [init_problem(shape) for _ in range(k)]

    # add folds to the right sub-problems
    for i, prob in enumerate(probs):
        for j, fold in enumerate(folds):
            if j == i:
                prob.test.extend(*fold)
            else:
                prob.train.extend(*fold)

    # compute alpha sequence if not provided
    if not alphas:
        alpha_maxs = [imputer.alpha_max(train) for imputer, train, _ in probs]

        alpha_max = max(alpha_maxs)
        alpha_min = alpha_max * alpha_min_ratio

        alphas = np.geomspace(alpha_max, alpha_min, n_alpha)

    # computing the test error for each fitted matrix
    sols = []
    for i, prob in enumerate(probs):
        print(f'fitting problem {i}')

        sols += [
            prob.fit_and_eval(alphas)
        ]

    # computing the total performance of each alpha
    total = sum(sol.errs for sol in sols)

    opt = np.argmin(total).item()
    est = sum(sol.mats[opt] for sol in sols) / len(sols)

    print(f'optimum = {opt}')
    print(f'optimum alpha = {alphas[opt]}')
    print(f'maximum alpha = {alphas[0]}')
    print(f'minimum alpha = {alphas[-1]}')

    return est, alphas[opt], sols


def __main__():
    npr.seed(4)

    shape = 500, 500
    n_row, n_col = shape

    rank = 3

    p = 0.02

    bl = npr.randn(n_row, rank)
    br = npr.randn(n_col, rank)
    b = bl @ br.T

    xs = []
    ys = []
    for i in range(n_row):
        for j in range(n_col):
            if npr.binomial(1, p) == 1:
                xs.append((i, j, 1))
                ys.append(b[i, j] + npr.randn())

    bh, *_ = cv_impute(shape, xs, ys)

    rel_err = compute_rel_error(b, bh)

    print(f'relative error = {rel_err}')


if __name__ == '__main__':
    __main__()
