"""The NGBoost LogNormalShift distribution and scores"""
import numpy as np
import scipy as sp
from scipy.stats import lognorm as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore


class LogNormalShiftLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((self.scale.shape[0], 3))
        D[:, 0] = (self.scale - lT) / (self.s ** 2)
        D[:, 1] = 1  #TODO: location 
        D[:, 2] = 1 - ((self.scale - lT) ** 2) / (self.s ** 2)
        return D

    def metric(self):
        FI = np.zeros((self.scale.shape[0], 3, 3))
        FI[:, 0, 0] = 1 / (self.s ** 2) + self.eps
        FI[:, 1, 1] = 1  #TODO: location
        FI[:, 2, 2] = 2
        return FI


class LogNormalShiftCRPScore(CRPScore):
    def score(self, Y):
        lY = np.log(Y)
        Z = (lY - self.scale) / self.s
        return self.s * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
        )

    def d_score(self, Y):
        lY = np.log(Y)
        Z = (lY - self.scale) / self.s

        D = np.zeros((self.scale.shape[0], 3))
        D[:, 0] = -(2 * sp.stats.norm.cdf(Z) - 1)
        D[:, 1] = 1  #TODO: location
        D[:, 2] = self.score(Y) + (lY - self.scale) * D[:, 0]
        return D

    def metric(self):
        I = np.zeros((self.scale.shape[0], 3, 3))
        I[:, 0, 0] = 2
        I[:, 1, 1] = 1  #TODO: location
        I[:, 2, 2] = self.s ** 2
        I /= 2 * np.sqrt(np.pi)
        return I


class LogNormalShift(RegressionDistn):

    """
    Implements the log-normal distribution with an extra shift 
    parameter for NGBoost.

    The log-normal distribution has two parameters, s and scale and location
    parameter that shifts the distribution (see scipy.stats.lognorm).
    This distribution has both LogScore and CRPScore implemented.
    """

    n_params = 3
    scores = [LogNormalShiftLogScore, LogNormalShiftCRPScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.s = np.exp(params[0])
        self.logs = params[0]
        self.loc = params[1]
        self.scale = np.exp(params[2])
        self.logscale = params[2]
        self.dist = dist(s=self.s, loc=self.loc, scale=self.scale)
        self.eps = 1e-5

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def fit(Y):
        s, loc, scale = sp.stats.lognorm.fit(Y)
        return np.array([np.log(s), loc, np.log(scale)])

    @property
    def params(self):
        return {"s": self.s, "loc": self.loc, "scale": self.scale}

