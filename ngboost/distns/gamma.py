"""The NGBoost Gamma distribution and scores"""
import numpy as np
import scipy as sp
from scipy.stats import gamma as dist
from scipy.special import polygamma

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore


class GammaLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        lY = np.log(Y)
        D = np.zeros((self.scale.shape[0], 2))
        D[:, 0] = self.shape*(polygamma(0, self.shape) + np.log(self.scale) - lY)
        D[:, 1] = self.shape - Y/self.scale
        return D

    def metric(self):
        FI = np.zeros((self.scale.shape[0], 2, 2))
        FI[:, 0, 0] = self.shape*(polygamma(1, self.shape)
        FI[:, 0, 1] = 
        FI[:, 1, 0] = FI[:, 0, 1]
        FI[:, 1, 1] = 2
        return FI


class Gamma(RegressionDistn):

    """
    Implements the log-normal distribution with an extra shift 
    parameter for NGBoost.

    The log-normal distribution has two parameters, s and scale and location
    parameter that shifts the distribution (see scipy.stats.lognorm).
    This distribution has both LogScore and CRPScore implemented.
    """

    n_params = 2
    scores = [GammaLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.shape = np.exp(params[0])
        self.scale = np.exp(params[1])
        self.loc = 0.0
        self.dist = dist(s=self.shape, scale=self.scale)
        self.eps = 1e-5

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def sample(self, m):
        return self.dist.rvs(size=m)

    def fit(Y):
        s, _, scale = dist.fit(Y, floc=self.loc)
        return np.array([np.log(s), np.log(scale)])

    @property
    def params(self):
        return {"s": self.shape, "scale": self.scale}

