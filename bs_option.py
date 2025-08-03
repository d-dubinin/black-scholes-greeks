import numpy as np
from scipy.stats import norm

class BlackScholesOption:
    def __init__(self, S_t, K, T, t, r, sigma, option_type='call'):
        """
        S_t : Spot price at time t
        K : Strike price
        T : Maturity (absolute time)
        t : Current time (0 <= t < T)
        r : Risk-free rate
        sigma : Volatility
        option_type : 'call' or 'put'
        """
        self.S_t = S_t
        self.K = K
        self.T = T
        self.t = t
        self.tau = T - t 
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.tau <= 0:
            raise ValueError("Time to maturity must be positive (T > t)")

    def _d1(self):
        return (np.log(self.S_t / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.tau) / (self.sigma * np.sqrt(self.tau))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.tau)

    def price(self):
        d1 = self._d1()
        d2 = self._d2()
        if self.option_type == 'call':
            return self.S_t * norm.cdf(d1) - self.K * np.exp(-self.r * self.tau) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S_t * norm.cdf(-d1)

    def delta(self):
        d1 = self._d1()
        return norm.cdf(d1) if self.option_type == 'call' else norm.cdf(d1) - 1

    def gamma(self):
        d1 = self._d1()
        return norm.pdf(d1) / (self.S_t * self.sigma * np.sqrt(self.tau))

    def vega(self):
        d1 = self._d1()
        return self.S_t * norm.pdf(d1) * np.sqrt(self.tau)

    def theta(self):
        d1 = self._d1()
        d2 = self._d2()
        first = - (self.S_t * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.tau))
        if self.option_type == 'call':
            second = - self.r * self.K * np.exp(-self.r * self.tau) * norm.cdf(d2)
        else:
            second = self.r * self.K * np.exp(-self.r * self.tau) * norm.cdf(-d2)
        return first + second