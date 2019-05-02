from typing import List, NamedTuple

import numpy as np
import numpy.random as npr

from impute.utils import SVD

from impute import Dataset
from impute import LagrangianImpute


# container classes
class CvProblem(NamedTuple):

    imputer: LagrangianImpute
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

        return CvSolution(svds, mats, errs)


class CvSolution(NamedTuple):
    svds: List[SVD]
    mats: List[np.ndarray]
    errs: np.ndarray


# evaluation helpers
def rss(y, yh) -> float:
    return np.sum((y - yh) ** 2).item()


def relative_error(b, bh):
    return rss(b, bh) / rss(b, 0)


# split and append helper functions
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


def add_to_subproblems(probs, folds):
    assert len(probs) == len(folds)

    for i, prob in enumerate(probs):
        for j, fold in enumerate(folds):
            if j == i:
                prob.test.extend(*fold)
            else:
                prob.train.extend(*fold)


def split_and_add_to_subproblems(probs, xs, ys, foldids=None, shuffle=True):
    assert len(xs) == len(ys)

    n = len(xs)
    k = len(probs)
    if not foldids:
        foldids = generate_foldids(n, k, shuffle)

    folds = [(subarray(xs, fold), subarray(ys, fold)) for fold in foldids]

    add_to_subproblems(probs, folds)


# cross-validation helper
def cv_impute(probs, n_alpha=100, alpha_min_ratio=0.01, alphas=None, verbose=1):

    # compute alpha sequence if not provided
    if not alphas:
        alpha_maxs = [imputer.alpha_max(train) for imputer, train, _ in probs]

        alpha_max = max(alpha_maxs)
        alpha_min = alpha_max * alpha_min_ratio

        alphas = np.geomspace(alpha_max, alpha_min, n_alpha)

    # computing the test error for each fitted matrix
    sols = []
    for i, prob in enumerate(probs):
        if verbose == 2:
            print(f'fitting problem {i}')

        sol = prob.fit_and_eval(alphas)
        sols += [sol]

        if verbose == 2:
            print(f'optimum[{i}] = {np.argmin(sol.errs)}')

    # computing the total performance of each alpha
    total = sum(sol.errs for sol in sols)

    opt = np.argmin(total).item()
    est = sum(sol.mats[opt] for sol in sols) / len(sols)

    if verbose:
        print(f'optimum = {opt}')
        print(f'optimum alpha = {alphas[opt]}')
        print(f'maximum alpha = {alphas[0]}')
        print(f'minimum alpha = {alphas[-1]}')

    return est, alphas[opt], sols


# miscellaneous
def subarray(arr, ind):
    return [arr[i] for i in ind]
