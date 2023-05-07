# This file is part of the Ektelo framework 
# For licensing terms please see: https://github.com/ektelo/ektelo
#
# Copyright 2019-2021, Tumult Labs Inc.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from functools import reduce
import math

class EkteloMatrix(LinearOperator):
    """
    An EkteloMatrix is a linear transformation that can compute matrix-vector products 
    """
    # must implement: _matmat, _transpose, matrix
    # can  implement: gram, sensitivity, sum, dense_matrix, sparse_matrix, __abs__

    def __init__(self, matrix):
        """ Instantiate an EkteloMatrix from an explicitly represented backing matrix
        
        :param matrix: a 2d numpy array or a scipy sparse matrix
        """
        self.matrix = matrix
        self.dtype = matrix.dtype
        self.shape = matrix.shape

    def _transpose(self):
        return EkteloMatrix(self.matrix.T)
    
    def _matmat(self, V):
        """
        Matrix multiplication of a m x n matrix Q
        
        :param V: a n x p numpy array
        :return Q*V: a m x p numpy aray
        """
        return self.matrix @ V

    def gram(self):
        """ 
        Compute the Gram matrix of the given matrix.
        For a matrix Q, the gram matrix is defined as Q^T Q
        """
        return self.T @ self # works for subclasses too
   
    def sensitivity(self):
        # note: this works because np.abs calls self.__abs__
        return np.max(np.abs(self).sum(axis=0))
 
    def sum(self, axis=None):
        # this implementation works for all subclasses too 
        # (as long as they define _matmat and _transpose)
        if axis == 0:
            return self.T.dot(np.ones(self.shape[0]))
        ans = self.dot(np.ones(self.shape[1]))  
        return ans if axis == 1 else np.sum(ans)

    def inv(self):
        return EkteloMatrix(np.linalg.inv(self.dense_matrix()))

    def pinv(self):
        return EkteloMatrix(np.linalg.pinv(self.dense_matrix()))

    def trace(self):
        return self.diag().sum()

    def diag(self):
        return np.diag(self.dense_matrix())

    def _adjoint(self):
        return self._transpose()

    def __mul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)
        if type(other) == np.ndarray:
            return self.dot(other)
        if isinstance(other, EkteloMatrix):
            return Product(self, other)
            # note: this expects both matrix types to be compatible (e.g., sparse and sparse)
            # todo: make it work for different backing representations
        else:
            raise TypeError('incompatible type %s for multiplication with EkteloMatrix'%type(other))

    def __add__(self, other):
        if np.isscalar(other):
            other = Weighted(Ones(self.shape), other)
        return Sum([self, other])

    def __sub__(self, other):
        return self + -1*other
            
    def __rmul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)
        return NotImplemented

    def __getitem__(self, key):
        """ 
        return a given row from the matrix
    
        :param key: the index of the row to return
        :return: a 1xN EkteloMatrix
        """
        # row indexing, subclasses may provide more efficient implementation
        m, n = self.shape
        v = np.zeros(m)
        v[key] = 1.0
        return EkteloMatrix(self.T.dot(v).reshape(1, n))
    
    def dense_matrix(self):
        """
        return the dense representation of this matrix, as a 2D numpy array
        """
        if sparse.issparse(self.matrix):
            return self.matrix.toarray()
        return self.matrix
    
    def sparse_matrix(self):
        """
        return the sparse representation of this matrix, as a scipy matrix
        """
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.matrix)
    
    @property
    def ndim(self):
        # todo: deprecate if possible
        return 2
    
    def __abs__(self):
        return EkteloMatrix(self.matrix.__abs__())
    
    def __sqr__(self):
        if sparse.issparse(self.matrix):
            return EkteloMatrix(self.matrix.power(2))
        return EkteloMatrix(self.matrix**2)

    def l1_sensitivity(self):
        return self.__abs__().sum(axis=0).max()

    def l2_sensitivity(self):
        return np.sqrt(self.__sqr__().sum(axis=0).max())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        return hash(repr(self))
   
class Identity(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (n,n)
        self.dtype = dtype
   
    def _matmat(self, V):
        return V
 
    def _transpose(self):
        return self

    @property
    def matrix(self):
        return sparse.eye(self.n, dtype=self.dtype)
    
    def __mul__(self, other):
        assert other.shape[0] == self.n, 'dimension mismatch'
        return other

    def inv(self):
        return self

    def pinv(self):
        return self

    def trace(self):
        return self.n

    def __abs__(self):  
        return self

    def __sqr__(self):
        return self


#####################################################################


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


####################################################################


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


####################################################################

import itertools
from mbi import Dataset, GraphicalModel, FactoredInference, Domain
from collections import defaultdict
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse


def powerset(iterable):
  "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))

def downward_closure(Ws):
  ans = set()
  for proj in Ws:
    ans.update(powerset(proj))
  return list(sorted(ans, key=len))

def hypothetical_model_size(domain, cliques):
  model = GraphicalModel(domain, cliques)
  return model.size * 8 / 2**20

def compile_workload(workload):
  def score(cl):
    return sum(len(set(cl)&set(ax)) for ax in workload)
  return { cl : score(cl) for cl in downward_closure(workload) }

def filter_candidates(candidates, model, size_limit):
  ans = { }
  free_cliques = downward_closure(model.cliques)
  for cl in candidates:
    cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
    cond2 = cl in free_cliques
    if cond1 or cond2:
      ans[cl] = candidates[cl]
  return ans


class AIM(Mechanism):
  def __init__(self, epsilon, delta, prng=None, rounds=None, max_model_size=80, structural_zeros={}):
    super(AIM, self).__init__(epsilon, delta, prng)
    self.rounds = rounds
    self.max_model_size = max_model_size
    self.structural_zeros = structural_zeros

  def worst_approximated(self, candidates, answers, model, eps, sigma):
    errors = {}
    sensitivity = {}
    for cl in candidates:
      wgt = candidates[cl]
      x = answers[cl]
      bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
      xest = model.project(cl).datavector()
      errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
      sensitivity[cl] = abs(wgt) 
    
    max_sensitivity = max(sensitivity.values()) # if all weights are 0, could be a problem
    return self.exponential_mechanism(errors, eps, max_sensitivity)

  def fit(self, data, W):    # what is W???
    rounds = self.rounds or 16*len(data.domain)
    workload = [cl for cl, _ in W]
    candidates = compile_workload(workload)
    answers = { cl : data.project(cl).datavector() for cl in candidates }

    oneway = [cl for cl in candidates if len(cl) == 1]

    sigma = np.sqrt(rounds / (2*0.9*self.rho))
    epsilon = np.sqrt(8*0.1*self.rho/rounds)
    
    measurements = []
    print('Initial Sigma', sigma)
    rho_used = len(oneway)*0.5/sigma**2
    for cl in oneway:
      x = data.project(cl).datavector()
      y = x + self.gaussian_noise(sigma,x.size)
      I = Identity(y.size) 
      measurements.append((I, y, sigma, cl))

    zeros = self.structural_zeros
    engine = FactoredInference(data.domain,iters=1000,warm_start=True,structural_zeros=zeros)
    model = engine.estimate(measurements)

    t = 0
    terminate = False
    while not terminate:
      t += 1
      if self.rho - rho_used < 2*(0.5/sigma**2 + 1.0/8 * epsilon**2):
        # Just use up whatever remaining budget there is for one last round
        remaining = self.rho - rho_used
        sigma = np.sqrt(1 / (2*0.9*remaining))
        epsilon = np.sqrt(8*0.1*remaining)
        terminate = True

      rho_used += 1.0/8 * epsilon**2 + 0.5/sigma**2
      size_limit = self.max_model_size*rho_used/self.rho

      small_candidates = filter_candidates(candidates, model, size_limit)
      cl = self.worst_approximated(small_candidates, answers, model, epsilon, sigma)

      n = data.domain.size(cl)
      Q = Identity(n) 
      x = data.project(cl).datavector()
      y = x + self.gaussian_noise(sigma, n)
      measurements.append((Q, y, sigma, cl))
      z = model.project(cl).datavector()

      model = engine.estimate(measurements)
      w = model.project(cl).datavector()
      print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
      if np.linalg.norm(w-z, 1) <= sigma*np.sqrt(2/np.pi)*n:
        print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma/2)
        sigma /= 2
        epsilon *= 2

    engine.iters = 2500
    model = engine.estimate(measurements)

    return model
  

  def generate(self, num_samples):
    print('Generating Data...')
    synth = self.model.synthetic_data(num_samples)

    return synth.df    

