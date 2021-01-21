"""The NGBoost SHASH distribution and scores"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import kv

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class SHASHLogScore(LogScore):
    """
    MLE score with metric (Fisher Information Matrix) calculated using
    random sampling defined in the parent class.
    """

    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        """Gradient of score"""
        z = (Y - self.loc) / self.scale
        r = self.nu - self.tau * np.arcsinh(z)
        w = np.sinh(r) * np.cosh(r) - np.tanh(r)
        v = 1./(z*z + 1.)
        u = self.tau * w * np.sqrt(v)

        D = np.zeros((self.scale.shape[0], 4))
        D[:, 0] = u/self.scale - z*z*v     
        D[:, 1] = z*u + v
        D[:, 2] = w
        D[:, 3] = -self.tau * np.arcsinh(z) * w - 1.
        return D


class SHASHo:
    def __init__(self, nu, tau, loc, scale):
        self.loc = loc
        self.scale = scale
        self.nu = nu
        self.tau = tau

    def logpdf(self, Y):
        """Negative loglikelihood for SHASH: follows implementation
        of dSHASHo function in R package gamlss."""
        z = (Y - self.loc)/self.scale
        w = self.tau * np.arcsinh(z) - self.nu
        s = np.sinh(w)
        c = np.sqrt(1 + s*s)
        return np.log(c*self.tau/self.scale) - 0.5*(np.log(2*np.pi*(1 + z*z)) + s*s)


class SHASH(RegressionDistn):
    """
    Implements the Sinh-ArcSinh (SHASH) distribution for NGBoost.

    The SHASH distribution defined in Jones & Pewsey (2009, p2)
    has four parameters: loc, scale, nu, and tau,
    defining the location, scale, skewness, and tail width.

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
        self.dist = SHASHo(self.nu, self.tau, loc=self.loc, scale=self.scale)

    def __getattr__(self, name):
        if name in dir(self):
            return getattr(self, name)
        return None

    def mean(self):
        q = 1/self.tau
        # Modified Bessel functions of the second kind
        Kp = kv(0.25*(q + 1), 0.25)  
        Km = kv(0.25*(q - 1), 0.25)
        P = np.exp(0.25)/np.sqrt(8*np.pi)*(Kp + Km)
        return self.loc + self.scale * np.sinh(self.nu*q) * P
    
    def sample(self, m):
        """Sample SHASH pdf: follows R gamlss rSHASHo"""
        z = np.random.randn(m, len(self.scale))  # Sample from standard normal
        x = self.loc + self.scale*np.sinh((np.arcsinh(z) + self.nu)/self.tau)
        return self.loc + self.scale*np.sinh((np.arcsinh(z) + self.nu)/self.tau)

    def logpdf(self, Y):
        """Negative loglikelihood for SHASH: follows implementation
        of dSHASHo function in R package gamlss."""
        z = (Y - self.loc)/self.scale
        w = self.tau * np.arcsinh(z) - self.nu
        s = np.sinh(w)
        c = np.sqrt(1 + s*s)
        return np.log(c*self.tau/self.scale) - 0.5*(np.log(2*np.pi*(1 + z*z)) + s*s)

    @staticmethod
    def dSHASHo(pars, Y):
        """loglikelihood for all points in Y"""
        loc, scale, nu, tau = pars
        scale = np.exp(scale)
        tau = np.exp(tau)

        z = (Y - loc)/scale
        w = tau * np.arcsinh(z) - nu
        r = np.sinh(w)
        c = np.sqrt(1 + r*r)
        ll = np.log(c*tau/scale) - 0.5*(np.log(2*np.pi*(1 + z*z)) + r*r)
        return np.exp(ll)

    @staticmethod
    def grad_dSHASHo(pars, Y):
        """Gradient of nll with the interface for scipy minimize
        Scale and nu parameters are converted from the internal logspace"""
        loc, scale, nu, tau = pars
        scale = np.exp(scale)
        tau = np.exp(tau)

        z = (Y - loc) / scale
        r = nu - tau * np.arcsinh(z)
        w = np.sinh(r) * np.cosh(r) - np.tanh(r)
        v = 1./(z*z + 1.)
        u = tau * w * np.sqrt(v)

        D = np.zeros((Y.shape[0], 4), dtype=np.float64)
        D[:, 0] = u/scale - z*z*v     
        D[:, 1] = z*u + v
        D[:, 2] = w
        D[:, 3] = -tau * np.arcsinh(z) * w - 1.
        return D

    @staticmethod
    def nll(pars, Y):
        """Negative loglikelihood with the interface for scipy minimize.
        Scale and nu parameters are converted from the internal logspace"""
        loc, scale, nu, tau = pars
        scale = np.exp(scale)
        tau = np.exp(tau)

        z = (Y - loc)/scale
        w = tau * np.arcsinh(z) - nu
        r = np.sinh(w)
        c = np.sqrt(1 + r*r)
        ll = np.log(c*tau/scale) - 0.5*(np.log(2*np.pi*(1 + z*z)) + r*r)
        return -np.mean(ll)

    @staticmethod
    def grad_nll(pars, Y):
        """Gradient of nll with the interface for scipy minimize
        Scale and nu parameters are converted from the internal logspace"""
        loc, scale, nu, tau = pars
        scale = np.exp(scale)
        tau = np.exp(tau)

        z = (Y - loc) / scale
        r = nu - tau * np.arcsinh(z)
        w = np.sinh(r) * np.cosh(r) - np.tanh(r)
        v = 1./(z*z + 1.)
        u = tau * w * np.sqrt(v)

        D = np.zeros((Y.shape[0], 4), dtype=np.float64)
        D[:, 0] = u/scale - z*z*v     
        D[:, 1] = z*u + v
        D[:, 2] = w
        D[:, 3] = -tau * np.arcsinh(z) * w - 1.
        return np.mean(D, axis=0)

    def fit(Y):
        """Minimize negative loglikelihood of the observations Y"""
        m, s = norm.fit(Y) # Get first estimate using normal distribution
        print("Fitted normal: ", m, s, np.log(s))
        pars0 = [m, np.log(s*1.1), 0., 0.0] 
        #res = minimize(SHASH.nll, pars0, args = (Y,), method = 'BFGS', jac = SHASH.grad_nll)
        res = minimize(SHASH.nll, pars0, args = (Y,), method = 'Nelder-Mead') #, jac = SHASH.grad_nll)
        print("result:", res)
        print("fitted", res.x[0], np.exp(res.x[1]), res.x[2], np.exp(res.x[3]))
        return res.x

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale,
                "nu": self.nu, "tau": self.tau}
