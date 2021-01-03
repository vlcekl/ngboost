"""The NGBoost SHASH distribution and scores"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class SHASHLogScore(LogScore):

    def score(self, Y):
        return -self.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((self.scale.shape[0], 4))
        D[:, 0] = #TODO 
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
        self.nu = params[2]
        self.tau = np.exp(params[3])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def sample(self, m):
        # Sample standard normal
        samples = np.random.randn(m)
        # Transform to SHASH
        samples = 
        return samples
    
    def sample2(self, m):
        # Sample uniform 
        samples = np.random.rand(m)
        # Transform to SHASH using inverse cummulative distribution
        #samples = 
        return samples

    @staticmethod
    def loglik(Y, loc, scale, nu, tau):
        z = (Y - loc)/(scale * tau)
        w = tau * np.arcsinh(z) - nu
        c = np.cosh(w)
        r = np.sinh(w)
        loglik = -np.log(scale) - 0.5*np.log(2*np.pi) - 0.5*np.log(1 + z*z) + np.log(c) - 0.5*r*r
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
