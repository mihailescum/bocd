# Copyright (c) 2014 Johannes Kulick
# Copyright (c) 2017-2018 David Tolpin
#
# Code initally borrowed from:
#    https://github.com/hildensia/bayesian_changepoint_detection and
#    https://github.com/dtolpin/bocd
# under the MIT license.


import numpy as np
from scipy import stats


class BayesianDetector:
    def __init__(self, hazard_function, observation_likelihood, tail_threshold):
        """Initializes the detector with zero observations.
        """
        self._t0 = 0
        self._t = -1
        self._growth_probs = np.array([1.0])
        self._hazard_function = hazard_function
        self._observation_likelihood = observation_likelihood
        self._tail_threshold = tail_threshold
        self._last_cp = 0

    def update(self, x):
        self._t += 1

        t = self._t - self._t0

        # allocate enough space
        if len(self._growth_probs) == t + 1:
            self._growth_probs = np.resize(self._growth_probs, (t + 1) * 2)

        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        pred_probs = self._observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = self._hazard_function(t + 1)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0
        self._growth_probs[self._growth_probs < 1e-10] = 0
        cp_prob = np.sum(self._growth_probs[0 : t + 1] * pred_probs * H)
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        self._growth_probs[1 : t + 2] = (
            self._growth_probs[0 : t + 1] * pred_probs * (1 - H)
        )
        # Put back changepoint probability
        self._growth_probs[0] = cp_prob
        self._growth_probs[t + 2 :] = 0

        # Renormalize the run length probabilities for improved numerical
        # stability.
        total_prob = np.sum(self._growth_probs[0 : t + 2])
        if total_prob >= 1e-15:
            self._growth_probs[0 : t + 2] = self._growth_probs[0 : t + 2] / total_prob
        else:
            #import pdb; pdb.set_trace()
            self._growth_probs[0] = 1
            self._growth_probs[1 : t + 2] = 0
            
        # Update the parameter sets for each possible run length.
        self._observation_likelihood.update_theta(x)

        cdf = np.cumsum(self._growth_probs[0 : t + 2])
        tail = np.flatnonzero(cdf > 1 - self._tail_threshold)
        if tail.size > 1:
            self.prune(tail[1] - 1)


    def get_growth_probabilities(self):
        return self._growth_probs.copy()
    
    def predictive_mean(self):
        MAP = np.argmax(self._growth_probs)
        return self._observation_likelihood.mu[MAP]
    
    def predictive_std(self):
        MAP = np.argmax(self._growth_probs)
        return self._observation_likelihood.get_scale()[MAP]


    def prune(self, t0):
        """prunes memory before time._t0. That is, pruning at t=0
        does not change the memory. One should prune at times
        which are likely to correspond to changepoints.
        """
        self._growth_probs[t0 + 1 :] = 0
        self._growth_probs[: t0 + 1] = self._growth_probs[: t0 + 1] / np.sum(
            self._growth_probs[: t0 + 1]
        )
        self._observation_likelihood.prune(t0)
        self._t0 = self._t - t0 + 1

    def detect(self, tail_slope, min_tail_probability):
        time_since_last_cp = self._t - self._last_cp
        tail = time_since_last_cp * tail_slope
        tail_probability = np.sum(self._growth_probs[int(tail):])
        if tail_probability < min_tail_probability:
            MAP = np.argmax(self._growth_probs)
            self._last_cp = self._t - MAP
            return MAP
        else:
            return -1


class StudentT:
    """Posterior if the Likelihood function is a Normal with
    unknown mean and variance and the (conjugate) prior a Normal-Inverse-Gamma.
    """

    def __init__(self, alpha, beta, kappa, mu):
        """ Mean of (normal) likelihood function was estimated from kappa observations
        with sample mean mu, variance was estimated from 2*alpha observations with 
        sample mean mu and sum of sqared deviations 2*beta.
        """
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

        self._size = 1

    def pdf(self, data):
        scale = self.get_scale()
        mask = np.abs(data - self.mu) < 5 * scale
        
        pdf = np.zeros(self.mu.size, dtype=float)
        pdf[:] = 1e-10
        pdf[mask] = stats.t.pdf(
            x=data,
            df=2 * self.alpha[mask],
            loc=self.mu[mask],
            scale=scale[mask],
        )
        return pdf
    
    def get_scale(self):
        return np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))

    def update_theta(self, data):
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

    def prune(self, t):
        """Prunes memory before t.
        """
        self.mu = self.mu[: t + 1]
        self.kappa = self.kappa[: t + 1]
        self.alpha = self.alpha[: t + 1]
        self.beta = self.beta[: t + 1]


class ConstantHazard:
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, r):
        """
        Args:
          r: The length of the current run (np.ndarray or scalar)
        Returns:
          p: Changepoint Probabilities(np.ndarray with shape = r.shape)
        """
        if isinstance(r, np.ndarray):
            r[:] = 1
        else:
            r = 1

        return r / self._lambda
