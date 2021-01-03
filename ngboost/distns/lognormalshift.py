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
        dY = Y - self.loc
        ldY = np.log(dY)
        D = np.zeros((self.scale.shape[0], 3))
        D[:, 0] = (self.mu - ldY) / self.sigma**2
        D[:, 1] = 1 - ((self.mu - ldY) ** 2) / (self.sigma ** 2)
        D[:, 2] = ((self.mu - ldY) / self.sigma**2 - 1) / dY
        return D

    #def metric(self):
    #    FI = np.zeros((self.scale.shape[0], 3, 3))
    #    FI[:, 0, 0] = 1 / (self.sigma ** 2) + self.eps
    #    FI[:, 1, 1] = 2
    #    FI[:, 2, 2] = 1  #TODO: location ugly
    #    return FI


class LogNormalShift(RegressionDistn):

    """
    Implements the log-normal distribution with an extra shift 
    parameter for NGBoost.

    The log-normal distribution has two parameters, s and scale and location
    parameter that shifts the distribution (see scipy.stats.lognorm).
    This distribution has both LogScore and CRPScore implemented.
    """

    n_params = 3
    scores = [LogNormalShiftLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.mu = params[0]             # mu of lognormal dist
        self.sigma = np.exp(params[1])  # sigma of lognormal dist
        self.loc = params[2]            # shift of the distribution origin
        self.dist = dist(s=self.sigma, loc=self.loc, scale=np.exp(self.mu))
        self.eps = 1e-5

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def sample(self, m):
        return self.dist.rvs(size=m)

    def fit(Y):
        s, loc, scale = sp.stats.lognorm.fit(Y)
        return np.array([np.log(scale), np.log(s), loc])

    @property
    def params(self):
        return {"s": self.sigma, "loc": self.loc, "scale": self.exp(self.mu)}

