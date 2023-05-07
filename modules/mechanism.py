import numpy as np
from autodp import privacy_calibrator
from functools import partial
#from cdp2adp import cdp_rho
from scipy.special import softmax


#Code for computing approximate differential privacy guarantees
# for discrete Gaussian and, more generally, concentrated DP
# See https://arxiv.org/abs/2004.00010
# - Thomas Steinke dgauss@thomas-steinke.net 2020

import math
import matplotlib.pyplot as plt

#*********************************************************************
#Now we move on to concentrated DP
    
#compute delta such that
#rho-CDP implies (eps,delta)-DP
#Note that adding cts or discrete N(0,sigma2) to sens-1 gives rho=1/(2*sigma2)

#start with standard P[privloss>eps] bound via markov
def cdp_delta_standard(rho,eps):
    assert rho>=0
    assert eps>=0
    if rho==0: return 0 #degenerate case
    #https://arxiv.org/pdf/1605.02065.pdf#page=15
    return math.exp(-((eps-rho)**2)/(4*rho))

#Our new bound:
# https://arxiv.org/pdf/2004.00010v3.pdf#page=13
def cdp_delta(rho,eps):
    assert rho>=0
    assert eps>=0
    if rho==0: return 0 #degenerate case

    #search for best alpha
    #Note that any alpha in (1,infty) yields a valid upper bound on delta
    # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
    # This code has two "hacks".
    # First the binary search is run for a pre-specificed length.
    # 1000 iterations should be sufficient to converge to a good solution.
    # Second we set a minimum value of alpha to avoid numerical stability issues.
    # Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit this constraint
    # when eps<=rho or close to it. This is not an interesting parameter regime, as you will
    # inherently get large delta in this regime.
    amin=1.01 #don't let alpha be too small, due to numerical stability
    amax=(eps+1)/(2*rho)+2
    for i in range(1000): #should be enough iterations
        alpha=(amin+amax)/2
        derivative = (2*alpha-1)*rho-eps+math.log1p(-1.0/alpha)
        if derivative<0:
            amin=alpha
        else:
            amax=alpha
    #now calculate delta
    delta = math.exp((alpha-1)*(alpha*rho-eps)+alpha*math.log1p(-1/alpha)) / (alpha-1.0)
    return min(delta,1.0) #delta<=1 always

#Above we compute delta given rho and eps, now we compute eps instead
#That is we wish to compute the smallest eps such that rho-CDP implies (eps,delta)-DP
def cdp_eps(rho,delta):
    assert rho>=0
    assert delta>0
    if delta>=1 or rho==0: return 0.0 #if delta>=1 or rho=0 then anything goes
    epsmin=0.0 #maintain cdp_delta(rho,eps)>=delta
    epsmax=rho+2*math.sqrt(rho*math.log(1/delta)) #maintain cdp_delta(rho,eps)<=delta
    #to compute epsmax we use the standard bound
    for i in range(1000):
        eps=(epsmin+epsmax)/2
        if cdp_delta(rho,eps)<=delta:
            epsmax=eps
        else:
            epsmin=eps
    return epsmax

#Now we compute rho
#Given (eps,delta) find the smallest rho such that rho-CDP implies (eps,delta)-DP
def cdp_rho(eps,delta):
    assert eps>=0
    assert delta>0
    if delta>=1: return 0.0 #if delta>=1 anything goes
    rhomin=0.0 #maintain cdp_delta(rho,eps)<=delta
    rhomax=eps+1 #maintain cdp_delta(rhomax,eps)>delta
    for i in range(1000):
        rho=(rhomin+rhomax)/2
        if cdp_delta(rho,eps)<=delta:
            rhomin=rho
        else:
            rhomax=rho
    return rhomin


##########################################################


def pareto_efficient(costs):
    eff = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if eff[i]:
            eff[eff] = np.any(costs[eff]<=c, axis=1)  # Keep any point with a lower cost
    return np.nonzero(eff)[0]

def generalized_em_scores(q, ds, t):
    q = -q
    idx = pareto_efficient(np.vstack([q, ds]).T)
    r = q + t*ds
    r = r[:,None] - r[idx][None,:]
    z = ds[:,None] + ds[idx][None,:]
    s = (r/z).max(axis=1)
    return -s

class Mechanism:
    def __init__(self, epsilon, delta, bounded, prng=np.random):
        """
        Base class for a mechanism.  
        :param epsilon: privacy parameter
        :param delta: privacy parameter
        :param bounded: privacy definition (bounded vs unbounded DP) 
        :param prng: pseudo random number generator
        """
        self.epsilon = epsilon
        self.delta = delta
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.bounded = bounded
        self.prng = prng

    def run(self, dataset, workload):
        pass

    def generalized_exponential_mechanism(self, qualities, sensitivities, epsilon, t=None, base_measure=None):
        if t is None:
            t = 2*np.log(len(qualities) / 0.5) / epsilon
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            sensitivities = np.array([sensitivities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            keys = np.arange(qualities.size)
        scores = generalized_em_scores(qualities, sensitivities, t)
        key = self.exponential_mechanism(scores, epsilon, 1.0, base_measure=base_measure)
        return keys[key]

    def permute_and_flip(self, qualities, epsilon, sensitivity=1.0):
        """ Sample a candidate from the permute-and-flip mechanism """
        q = qualities - qualities.max()
        p = np.exp(0.5*epsilon/sensitivity*q)
        for i in np.random.permutation(p.size):
            if np.random.rand() <= p[i]:
                return i

    def exponential_mechanism(self, qualities, epsilon, sensitivity=1.0, base_measure=None):
        if isinstance(qualities, dict):
            #import pandas as pd
            #print(pd.Series(list(qualities.values()), list(qualities.keys())).sort_values().tail())
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        """ Sample a candidate from the permute-and-flip mechanism """
        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5*epsilon/sensitivity*q)
        else:
            p = softmax(0.5*epsilon/sensitivity*q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    def gaussian_noise_scale(self, l2_sensitivity, epsilon, delta):
        """ Return the Gaussian noise necessary to attain (epsilon, delta)-DP """
        if self.bounded: l2_sensitivity *= 2.0
        return l2_sensitivity * privacy_calibrator.ana_gaussian_mech(epsilon, delta)['sigma']

    def laplace_noise_scale(self, l1_sensitivity, epsilon):
        """ Return the Laplace noise necessary to attain epsilon-DP """
        if self.bounded: l1_sensitivity *= 2.0
        return l1_sensitivity / epsilon

    def gaussian_noise(self, sigma, size):
        """ Generate iid Gaussian noise  of a given scale and size """
        return self.prng.normal(0, sigma, size)

    def laplace_noise(self, b, size):
        """ Generate iid Laplace noise  of a given scale and size """
        return self.prng.laplace(0, b, size)

    def best_noise_distribution(self, l1_sensitivity, l2_sensitivity, epsilon, delta):
        """ Adaptively determine if Laplace or Gaussian noise will be better, and
            return a function that samples from the appropriate distribution """
        b = self.laplace_noise_scale(l1_sensitivity, epsilon)
        sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
        dist = self.gaussian_noise if np.sqrt(2)*b > sigma else self.laplace_noise
        if np.sqrt(2)*b < sigma:
            return partial(self.laplace_noise, b)
        return partial(self.gaussian_noise, sigma)


