from __future__ import division, print_function
# import matplotlib.pyplot as plt 
import numpy as np
from numpy import log
import time
# import seaborn as sb
import scipy.stats as stats
from scipy.special import logsumexp
import particles 
from particles import distributions as dists
from particles import collectors 
import numpy as np
from scipy import stats
from scipy.linalg import cholesky, LinAlgError
import particles
from particles import smc_samplers as ssp
from particles import utils
import pandas as pd


# Seed
rng = np.random.default_rng(seed = 42) 
def stepX(xp, 
          eta, alpha, zeta, beta, beta_0, 
          gamma_u, gamma_d, lambda_u, lambda_d,
          alpha_s, kappa_s):
    """
    Performs a step of the kernel
    """
    # xp[0] is u_t
    # xp[1] is s_t
    # The same holds for x
    
    x = np.zeros(shape = (2))
    
    # Computations for u_t
    uniu = dists.Uniform(a = 0., b = 1.).rvs(size = 1) # Uniform to "choose a move"
    
    if xp[0] == 0 :
        if uniu <= 1 - eta:
            # Dirac on xp = 0 
            x[0] = 0
        else:
            # Exponential law with rate zeta 
            x[0] = rng.exponential(scale = 1 / zeta, size = 1)
            
    elif xp[0] !=0:
        if uniu <= alpha:
            # Dirac on xp
            x[0] = xp[0]

        elif uniu <= alpha + beta:
            # Exponential law with rate zeta 
            x[0] = rng.exponential(scale = 1 / zeta, size = 1)

        elif uniu <= alpha + beta + beta_0:
            # Dirac on 0
            x[0] = 0

        elif uniu <= alpha + beta + beta_0 + gamma_u:
            # Drift upward (rate lambda_u/xp[0])
            x[0] = xp[0] + rng.exponential(scale = xp[0]/lambda_u, size = 1)

        else:
            # Drift downward (rate lambda_d/xp[0])
            x[0] = xp[0] - rng.exponential(scale = xp[0]/lambda_d, size = 1)
        
        
    # Computations for s_t
    unis = dists.Uniform(a = 0., b = 1.).rvs(size = 1) # Uniform to "choose" a move 
               
    if unis <= alpha_s:
        x[1] = xp[1]
    else:
        #x[1] = dists.Gamma(a = kappa_s, b = 1/kappa_s).rvs(size = 1)
        x[1] = dists.Gamma(a = kappa_s, b = kappa_s).rvs(size = 1)
    
    if x[0] <  0:
        x[0] = 0.
        
        
    return x 

def sampleY(x, kappa, theta, epsilon_b, epsilon_0, b):
    """
    Samples an observation y_t based on the vector x_t and some parameters
    Uses the 1st expression of the emission model 
    """
    
    # Sample the number of molecules initially sampled (x_t)
    x_t = dists.Poisson(rate = x[0]*x[1]/(kappa*theta)).rvs(size = 1)
                           
    # Sample the amplificaiton coefficient (a_t)
    #a_t = dists.Gamma(kappa, theta).rvs(size = 1)
    a_t = dists.Gamma(kappa, 1/theta).rvs(size = 1)
                       
    # Sample the read counts (y_t)
    uniy = dists.Uniform(a = 0., b = 1.).rvs(size = 1) # Uniform to "choose a move"
    
    # With proba "1 - epsilon_b - epsilon_0"
    if uniy < 1 - epsilon_b - epsilon_0:
        y = dists.Poisson(rate = x_t*a_t).rvs(size = 1)
    
    # With proba "epsilon_b"
    elif uniy < 1 - epsilon_0:
        # Generate following a Poisson distribution truncated in 0     
        # We generate from it until we get a non-zero value
        proposition = 0 
        while proposition == 0:
            proposition = dists.Poisson(rate = 1*a_t).rvs(size = 1)
        y = proposition
                       
    # With proba "epsilon_0"
    else:
        y = dists.DiscreteUniform(0, b+1) 
    return y 

def generate_data(length, 
                  eta, alpha, zeta, beta, beta_0, 
                  gamma_u, gamma_d, lambda_u, lambda_d,
                  alpha_s, kappa_s,
                  kappa, theta, epsilon_0, epsilon_b, b):
    """
    Function that allows to generate data (that we can then analyze)
    length  is the length of the data to be generated
    """
    
    # Initialisation 
    X = np.zeros(shape = (length, 2))
    Y = np.zeros(shape = length)
    
    # Time = 0
    X[0, 1] = 0. # Initial law chosen to be 0 for s_t
    X[0, 0] = 0.
    #X[0,0] = rng.exponential(scale = 1 / zeta, size = 1) #Non-zero initial law for u_t
    #X[0,0] = 0.15
    
    Y[0] = sampleY(X[0,:], kappa = kappa, theta = theta, epsilon_b = epsilon_b, epsilon_0 = epsilon_0, b = b)
    
    # Time = 1, ...., length; technically in our case it is space, not time 
    for time in range(1, length):
        
        # Update X (one kernel step)
        X[time, :] = stepX(X[time - 1, :], 
                        eta = eta, alpha = alpha, zeta = zeta, beta = beta, beta_0 = beta_0, 
                        gamma_u = gamma_u, gamma_d=gamma_d, lambda_u = lambda_u, lambda_d = lambda_d,
                        alpha_s = alpha_s, kappa_s = kappa_s)
    
        # Sample from Y|X (from the law og Y given X)
        Y[time] = sampleY(X[time,:], kappa = kappa, theta = theta, epsilon_b = epsilon_b, epsilon_0 = epsilon_0, b = b)
    
    return X, Y

class RnaProb(particles.FeynmanKac):
    
    def __init__(self, 
                 data_observed,
                 T,
                 sum_truncation = 10,
                 eta = 0.05, alpha=0.72, zeta=1.18, beta=8e-4+0.25, beta_0=6e-4, # eta = 7.2e-4
                 gamma_u=0.011, gamma_d=0.013, lambda_u=5.0, lambda_d=5.0,
                 alpha_s=0.53, kappa_s=0.31,
                 kappa=0.61, theta=1.93, epsilon_0=2e-6, epsilon_b=6.7e-4, b = 10):
        
        # The observed data ie the y_t
        self.data_observed = data_observed
        # The length of the data
        self.T = T
        # The truncation for the infinite sum in the likelihood
        self.sum_truncation = sum_truncation
        
        self.eta = eta 
        self.alpha = alpha
        self.zeta = zeta 
        self.beta = beta 
        self.beta_0 = beta_0 
        
        self.gamma_u = gamma_u 
        self.gamma_d = gamma_d 
        self.lambda_u = lambda_d 
        self.lambda_d = lambda_d
        
        self.alpha_s = alpha_s 
        self.kappa_s = kappa_s
        
        self.kappa = kappa 
        self.theta = theta 
        self.epsilon_0 = epsilon_0
        self.epsilon_b = epsilon_b 
        self.b = b 

    #@njit     
    def M0(self, N): # N particles 
        x0 = np.zeros(shape = (N,2)) 
        #x0[:,0] = rng.exponential(scale = 1 / self.zeta, size = N) #Non-zero initial law for u_t
        #x0[:,0] = 0.15
        return x0
    #@njit 
    def M(self, t, xp):
                 
        # Initialize empty vector
        # It has two columns, one for u_t and one for s_t
        x = np.zeros(shape = xp.shape) 
        N = xp.shape[0]
        
        # Loop over the particles
        for i in range(N):
            # Call the function that does one move using the kernel of x
            x[i,:] = stepX(xp[i,:], 
                          eta = self.eta, alpha = self.alpha, zeta = self.zeta, beta = self.beta, beta_0 = self.beta_0, 
                          gamma_u = self.gamma_u, gamma_d = self.gamma_d, lambda_u = self.lambda_u, lambda_d = self.lambda_d,
                          alpha_s = self.alpha_s, kappa_s = self.kappa_s) 
            
        return x
    
    #@njit             
    def logG(self, t, xp, x):
        
        # We wish to compute the log of the conditional likelihood, which involves 3 terms
        # We compute the log of each term, then we use the logsumexp funciton
        # In here, data_observed[t] corresponds to y_t
        
        # AND we wish to do so for each of the N particles, hence the looop 
        # MAYBE vectorization would be possible ? 
        
        N = x.shape[0] # The number of particles is the first dimension of the array x
        NlogGs = np.zeros(N) #Initizaling the empty vector of log probabilities
        
        # Computaion of the terms that do not depend on the particle (ie on the X)
        ## 2nd term - NB truncated in 0 (does not depend on the particle)
        ## The likelihood can be obtained by dividing the classical likelihood by 1 - probability of getting 0 
        # Beware the formulations of the NB. Given in the article they state its mean is rp/(1-p)
        # and in scipy it says r(1 -p')/p', we will use p' = 1-p

        term_2 = log(self.epsilon_b) + \
            stats.nbinom.logpmf(k = self.data_observed[t], n = self.kappa, p = 1 - (self.theta /(1+self.theta))) + \
            stats.nbinom.logpmf(k = 0, n = self.kappa, p = 1 - (self.theta /(1+self.theta)))

        ## 3rd term - Discrete Uniform 
        # Discarding this if statement would induce a small approximation but could greatly accelerate the computation
        term_3 = 0 
        if self.data_observed[t] <= self.b: # To account for the indicator in the density of the uniform
            term_3 = log(self.epsilon_0) + log(1/(self.b))

        
        for i in range(N):
            
            # 1st term - Truncated infinite sum
            # Computation of the log of the terms in the sum
            summation_terms = np.zeros(self.sum_truncation)
            
            # The loop involves a product that stays the same in the loop for one particle: u_t*s_t
            # we don't want to compute it  at each step
            product = x[i,0]*x[i,1]
            
            for x_t in range(0, self.sum_truncation):
                summation_terms[x_t] =  stats.poisson.logpmf(k = x_t, mu=product/(self.kappa * self.theta)) + \
                    stats.nbinom.logpmf(k = self.data_observed[t], n = self.kappa, p = 1 - (self.theta*x_t /(1+self.theta*x_t)))

            # Computation of the log of the sum (+ the log of the coefficient in front)
            term_1 = log(1 - self.epsilon_b - self.epsilon_0) + logsumexp(summation_terms)
            
            # Agregation of the different terms 
            NlogGs[i] = logsumexp(np.array([term_1, term_2, term_3]))
        
        return NlogGs


# Actual generation of data 
T = 1000
# # We use the same parameters that inside the Feynman-Kac models 

# data_X, data_Y  = generate_data(length = T, 
#               eta =0.05, alpha=0.72, zeta=1.18, beta=8e-4+0.25, beta_0=6e-4, #eta = 7.2e-4
#               gamma_u=0.011, gamma_d=0.013, lambda_u=5.0, lambda_d=5.0,
#               alpha_s=0.53, kappa_s=0.31,
#               kappa=0.61, theta=1.93, epsilon_0=2e-6, epsilon_b=6.7e-4, b = 10)

##########################################################################################
##########################################################################################

data_x = pd.read_csv('data/data_X.csv').drop(columns='Unnamed: 0')
data_y = pd.read_csv('data/data_Y.csv').drop(columns='Unnamed: 0')

data_X = np.array(pd.read_csv('data/data_X.csv').drop(columns='Unnamed: 0'))
data_Y = np.array(pd.read_csv('data/data_Y.csv').drop(columns='Unnamed: 0').T)[0]


print('Data loaded !')
'''
simulations = {}
# Preparation
for trace in range(10):
    N = 100
    fk_rna = RnaProb(data_observed = data_Y, T = T)
    alg = particles.SMC(fk = fk_rna, N = N, collect = [collectors.Moments()])
    # Actual runing of the algorithm
    start = time.time()
    alg.run()
    end = time.time()
    simulations[trace] = alg.summaries.moments
    print("Similation n°{} done ".format(trace))
    print("Computation time for data of length {} and {} particules: {} secondes".format(T, N, np.round(end - start,2)))



# plt.plot([m["mean"][0] for m in alg.summaries.moments], label = "filtered u_t")
# plt.plot(data_X[:,0], label = "true u_t")
# plt.legend()

print('ouin ouin ')

import plotly.graph_objects as go 
simulation_mean = pd.DataFrame()
for trace in range(10) : 
    simulation_mean[trace] = [m["mean"][0] for m in simulations[trace]]


fig = go.Figure()
fig.add_trace(go.Scatter(x=[*range(T)],y=data_x['0'],name='True u_t'))
for trace in range(10) : 
    simulation_mean[trace] = [m["mean"][0] for m in simulations[trace]]
    fig.add_trace(go.Scatter(x=[*range(T)] , y=[m["mean"][0] for m in simulations[trace]],name =f"filtered u_t of simu n° {trace}"))
fig.show()

print('All is ready !')

simulation_mean.to_csv('/Users/theoalegretti/Documents/GitHub/HMMC-project/data/simulation_10_mean_bootstrap.csv')
'''


"""
'alpha': 0.97
'gamma_u': 0.011
'gamma_d': 0.013
'beta': 8e-4
'beta_0':6e-4
'epsilon_0':2e-6
"""

prior_dict = {'eta': dists.Beta(1.,100.),
              'zeta': dists.Gamma(1,1)}
my_prior = dists.StructDist(prior_dict)


def msjd(theta):
    """Mean squared jumping distance.

    Parameters
    ----------
    theta: structured array

    Returns
    -------
    float
    """
    s = 0.
    for p in theta.dtype.names:
        s += np.sum(np.diff(theta[p], axis=0) ** 2)
    return s


class MCMC(object):
    """MCMC base class.

    To subclass MCMC, define methods:
        * `step0(self)`: initial step
        * `step(self, n)`: n-th step, n>=1

    """
    def __init__(self, niter=10, verbose=0):
        """
        Parameters
        ----------
        niter: int
            number of MCMC iterations
        verbose: int (default=0)
            progress report printed every (niter/verbose) iterations (never if 0)
        """
        self.niter = niter
        self.verbose = verbose

    def step0(self):
        raise NotImplementedError

    def step(self, n):
        raise NotImplementedError

    def mean_sq_jump_dist(self, discard_frac=0.1):
        """Mean squared jumping distance estimated from chain.

        Parameters
        ----------
        discard_frac: float
            fraction of iterations to discard at the beginning (as a burn-in)

        Returns
        -------
        float
        """
        discard = int(self.niter * discard_frac)
        return msjd(self.chain.theta[discard:])

    def print_progress(self, n):
        params = self.chain.theta.dtype.fields.keys()
        msg = 'Iteration %i' % n
        if hasattr(self, 'nacc') and n > 0:
            msg += ', acc. rate=%.3f' % (self.nacc / n)
        for p in params:
            msg += ', %s=%s' % (p, self.chain.theta[p][n])
        print(msg)

    @utils.timer
    def run(self):
        for n in range(self.niter):
            if n == 0:
                self.step0()
            else:
                self.step(n)
            if self.verbose > 0 and (n * self.verbose) % self.niter == 0:
                self.print_progress(n)

##################################
# Random walk Metropolis samplers

class VanishCovTracker(object):
    """Tracks the vanishing mean and covariance of a sequence of points.

    Computes running mean and covariance of points
    t^(-alpha) * X_t
    for some alpha \in [0,1] (typically)
    """
    def __init__(self, alpha=0.6, dim=1, mu0=None, Sigma0=None):
        self.alpha = alpha
        self.t = 0
        self.mu = np.zeros(dim) if mu0 is None else mu0
        if Sigma0 is None:
            self.Sigma = np.eye(dim)
            self.L0 = np.eye(dim)
        else:
            self.Sigma = Sigma0
            self.L0 = cholesky(Sigma0, lower=True)
        self.L = self.L0.copy()

    def gamma(self):
        return (self.t + 1)**(-self.alpha)  # not t, otherwise gamma(1)=1.

    def update(self, v):
        """Adds point v"""
        self.t += 1
        g = self.gamma()
        self.mu = (1. - g) * self.mu + g * v
        mv = v - self.mu
        self.Sigma = ((1. - g) * self.Sigma
                      + g * np.dot(mv[:, np.newaxis], mv[np.newaxis, :]))
        try:
            self.L = cholesky(self.Sigma, lower=True)
        except LinAlgError:
            self.L = self.L0

class GenericRWHM(MCMC):
    """Base class for random walk Hasting-Metropolis samplers.

    must be subclassed; the subclass must provide attribute self.prior
    """
    def __init__(self, niter=10, verbose=0, theta0=None,
                 adaptive=True, scale=1., rw_cov=None):
        """
        Parameters
        ----------

        niter: int
            number of MCMC iterations
        verbose: int (default=0)
            progress report printed every (niter/verbose) iterations (never if 0)
        theta0: structured array of size=1 or None
            starting point, simulated from the prior if set to None
        adaptive: True/False
            whether to use the adaptive version or not
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38^2 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array
            covariance matrix of the random walk proposal (set to I_d if None)
        """
        for k in ['niter', 'verbose', 'theta0', 'adaptive']:
            setattr(self, k, locals()[k])
        self.chain = ssp.ThetaParticles(
                        theta=np.empty(shape=niter, dtype=self.prior.dtype),
                        lpost=np.empty(shape=niter))
        self.nacc = 0
        self.arr = ssp.view_2d_array(self.chain.theta)
        self.dim = self.arr.shape[-1]
        if self.adaptive:
            optim_scale = 2.38 / np.sqrt(self.dim)
            self.scale = scale * optim_scale
            self.cov_tracker = VanishCovTracker(dim=self.dim, Sigma0=rw_cov)
            self.L = self.scale * self.cov_tracker.L
        else:
            if rw_cov is None:
                self.L = np.eye(self.dim)
            else:
                self.L = cholesky(rw_cov, lower=True)

    def step0(self):
        th0 = self.prior.rvs(size=1) if self.theta0 is None else self.theta0
        self.prop = ssp.ThetaParticles(theta=th0, lpost=np.zeros(1))
        self.prop_arr = ssp.view_2d_array(th0)
        self.compute_post()
        self.chain.copyto_at(0, self.prop, 0)

    def compute_post(self):
        """Computes posterior density at point self.prop"""
        raise NotImplementedError

    def step(self, n):
        z = stats.norm.rvs(size=self.dim)
        self.prop_arr[0] = self.arr[n - 1] + np.dot(self.L, z)
        self.compute_post()
        lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
        if np.log(stats.uniform.rvs()) < lp_acc:  # accept
            self.chain.copyto_at(n, self.prop, 0)
            self.nacc += 1
        else:  # reject
            self.chain.copyto_at(n, self.chain, n - 1)
        if self.adaptive:
            self.cov_tracker.update(self.arr[n])
            self.L = self.scale * self.cov_tracker.L

    @property
    def acc_rate(self):
        return self.nacc / (self.chain.N - 1)


class PMMH(GenericRWHM):
    """
    Particle Marginal Metropolis Hastings.
    PMMH is class of Metropolis samplers where the intractable likelihood of
    the considered state-space model is replaced by an estimate obtained from
    a particle filter.
    """

    def __init__(self, niter=10, verbose=0,fk = None,T = 0, 
                 smc_cls=particles.SMC, prior=None, data=None, smc_options=None, Nx=100, theta0=None, adaptive=True, scale=1.,
                 rw_cov=None):
        """
        Parameters
        ----------
        niter: int
            number of iterations
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        ssm_cls: StateSpaceModel class
            the considered parametric class of state-space models
        smc_cls: class (default: particles.SMC)
            SMC class
        prior: StructDist
            the prior
        data: list-like
            the data
        smc_options: dict
            options to pass to class SMC
        fk_cls: (default=Bootstrap)
            FeynmanKac class associated to the model
        Nx: int
            number of particles (for the particle filter that evaluates the
            likelihood)
        theta0: structured array of length=1
            starting point (generated from prior if =None)
        adaptive: bool
            whether to use the adaptive version
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array
            covariance matrix of the random walk proposal (set to I_d if None)
        """
        self.fk = fk 
        self.T = T 
        self.smc_cls = smc_cls
        self.prior = prior
        self.data = data
        # do not collect summaries, no need
        self.smc_options = {'collect': 'off'}
        if smc_options is not None:
            self.smc_options.update(smc_options)
        self.Nx = Nx
        GenericRWHM.__init__(self, niter=niter, verbose=verbose,
                             theta0=theta0, adaptive=adaptive, scale=scale,
                             rw_cov=rw_cov)

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk(data_observed = self.data,T = self.T,eta=theta['eta'],zeta=theta['zeta']),
                            N=self.Nx, **self.smc_options)

    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.prop.lpost[0] += pf.logLt


import plotly.graph_objects as go 
Nx = 20
niter = 1000

begin = time.time()

print(f"The PMMH running for {niter} iter and {Nx} particles  ")

mod = PMMH(fk=RnaProb,prior=my_prior, data=data_Y, Nx=Nx,niter=niter,T=T)


mod.run()


print(f"The PMMH runs in {np.round(time.time()-begin,2)} ' s ")

"""
for p in prior_dict.keys():  # loop over parameters involved in the bayesian inference
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[*range(niter)],y=mod.chain.theta[p]))
    fig.update_layout(
    title={
        'text': p,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
"""

df = pd.DataFrame(mod.chain.theta)
df['lpost']=mod.chain.lpost
df.to_csv(f"data/PMMH_iter-{niter}_particles-{Nx}.csv")

print(f"That's better ?  Here we have {niter} iterations and  {10} particles ")
print(f"In case ")
