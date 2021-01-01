"""The NGBoost SHASH distribution and scores"""
import numpy as np
from scipy.optimize import minimize

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class SHASHLogScore(LogScore):

    def score(self, Y):
        return -self.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((self.scale.shape[0], 4))
        D[:, 0] = (self.scale - lT) / (self.s ** 2)
        D[:, 1] = #TODO
        D[:, 2] = #TODO
        D[:, 3] = #TODO
        return D


class SHASH(RegressionDistn):

    """
    Implements the Sinh-ArcSinh (SHASH) distribution for NGBoost.

    The SHASH distribution has four parameters: loc, scale, nu, and tau,
    defining the location, scale, skew, and tail width.
    This distribution has LogScore implemented using the default metric method.
    """

    n_params = 4
    scores = [SHASHLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.logscale = params[1]
        self.nu = params[2]
        self.tau = np.exp(params[3])
        self.logtau = params[3]

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    @staticmethod
    def loglik(Y, mu, sigma, nu, tau):
        z = (Y - mu)/(sigma * tau)
        c = np.cosh(tau * np.arcsinh(z) - nu)
        r = np.sinh(tau * np.arcsinh(z) - nu)
        loglik = -np.log(sigma) - 0.5*np.log(2*np.pi) - 0.5*np.log(1 + (z*z)) + np.log(c) - 0.5*(r*r)
        return loglik

    def logpdf(self, Y):
        return self.loglik(Y, self.loc, self.scale, self.nu, self.tau)

    def fit(Y):
        """Minimize negative loglikelihood of the observations Y"""
        loc, scale, nu, tau = (0, 1, 0, 1)
        return np.array([loc, np.log(scale), nu, np.log(tau)])

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale,
                "nu": self.nu, "tau": self.tau)}
