import numpy as np 
from scipy.stats import norm

name = 'tiboT'

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first parameters are mu and beta, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity 
    N,K = x.shape

    xb_s = x@b / sig
    Phi = norm.cdf(xb_s)

    u_s = (y - x@b)/sig
    phi = norm.pdf(u_s) / sig

    # avoid taking log of zero
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)

    # loglikelihood function 
    ll = (y == 0.0) * np.log(Phi) + (y < 0) * np.log(phi)

    return ll

def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = x.shape 
    b_ols = np.linalg.solve(x.T@x, x.T@y)
    res = y - x@b_ols 
    sig2hat = 1./(N-K) * np.dot(res, res)
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

