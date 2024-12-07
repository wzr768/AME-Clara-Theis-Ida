import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import optimize
from scipy.stats import norm
from scipy.stats import t
from tabulate import tabulate
import estimation as est
import pandas as pd

name = 'Probit'

# global flag to make silly checks 
# disable to increase speed 
DOCHECKS = True 

def G(z): 
    return norm.cdf(z)

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x):

    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.shape[0] == N
        assert theta.ndim == 1 
        assert theta.size == K 

    z = x@theta
    Gxb = G(z)
    
    # we cannot take the log of 0.0
    Gxb = np.fmax(Gxb, 1e-8)    # truncate below at 1e-8 
    Gxb = np.fmin(Gxb, 1.-1e-8) # truncate above at 0.99999999

    ll = (y==1)*np.log(Gxb) + (y==0)*np.log(1.0 - Gxb) 
    return ll

def Ginv(p): 
    '''Inverse cdf
    Args. 
        p: N-array of values in [0;1] (probabilities)
    Returns
        x: N-array of values in (-inf; inf) 
    '''
    return norm.ppf(p)

def starting_values(y,x): 
    b_ols = np.linalg.solve(x.T@x, x.T@y)
    return b_ols*2.5

def predict(theta, x): 
    # the "prediction" is just Pr(y=1|x)
    yhat = G(x@theta) 
    return yhat 

def sim_data(theta: np.ndarray, N:int) -> tuple: 
    '''sim_data: simulate a dataset of size N with true K-parameter theta

    Args. 
        theta: (K,) vector of true parameters (k=0 will always be a constant)
        N (int): number of observations to simulate 
    
    Returns
        tuple: y,x
            y (float): binary outcome taking values 0.0 and 1.0
            x: (N,K) matrix of explanatory variables
    '''

    # 0. unpack parameters from theta
    # (simple, we are only estimating beta coefficients)
    beta = theta

    K = theta.size 
    assert K>1, f'Not implemented for constant-only'
    
    # 1. simulate x variables, adding a constant 
    oo = np.ones((N,1))
    xx = np.random.normal(size=(N,K-1))
    x = np.hstack([oo, xx]);
    
    # 2. simulate y values
    
    # 2.a draw error terms 
    uniforms = np.random.uniform(size=(N,))
    u = Ginv(uniforms)

    # 2.b compute latent index 
    ystar = x@beta + u
    
    # 2.b compute observed y (as a float)
    y = (ystar>=0).astype(float)

    # 3. return 
    return y, x

def compute_ape(thetahat, x, index):
    """
    Compute the Average Partial Effect (APE) on the probability of experiencing force.

    Parameters:
    - thetahat: A numpy array of estimated coefficients.
    - x: A numpy array of explanatory variables.
    - index: index of the regressor we want to calculate the average partial effect of

    Returns:
    - ape: The Average Partial Effect of the regressor we are considering.
    """

    # Number of observations
    N = x.shape[0]

    # Compute the baseline probabilities
    x_baseline = x.copy()
    x_baseline[:, 1:3] = 0 #white baseline
    baseline_probs = predict(thetahat, x_baseline)

    # Compute the counterfactual probabilities 
    x_counterfactual = x_baseline.copy()
    x_counterfactual[:, index] = 1  
    counterprobs = predict(thetahat, x_counterfactual)

    # Compute the individual-level difference in probabilities
    prob_differences = counterprobs - baseline_probs

    # Compute Average Partial Effect
    ape = np.mean(prob_differences)

    return ape

def properties(x, thetahat, print_out: bool, se: bool, indices, labels):
    """
    Compute various properties and statistics for a given dataset and estimated parameters for multiple regressors.
    
    Parameters:
    - x (numpy.ndarray): 2D array representing the dataset with dimensions (N, K),
                        where N is the number of observations, and K is the number of parameters.
    - thetahat (numpy.ndarray): Estimated parameters for the model.
    - print_out (bool): If True, print the results as a DataFrame.
    - indices (list): List of indices corresponding to the regressors we want to calculate the APE for.
    - labels (list): List of labels corresponding to each regressor.
    
    Returns:
    - If print_out is True, returns a DataFrame containing estimates, standard errors,
      t-values, and p-values for various model properties.
    - If print_out is False, returns a numpy.ndarray containing the same information.
    """

    # Initialize lists to store the results
    ape_list = []

    # Loop through the indices to compute the APE for each regressor
    for index in indices:
        ape = compute_ape(thetahat, x, index)
        ape_list.append(ape)

    # Organize the results into a DataFrame if `print_out` is True
    if print_out:
        # Create a DataFrame with the results and use the labels as the index
        data = {
            'Estimate': ape_list,
        }
        df = pd.DataFrame(data, index=labels)  # Use labels for the index
        df = df.round(3)  # Round the results to 4 decimal places
        return df
    else:
        # If `print_out` is False, return the raw data
        return {
            'Estimate': ape_list,
        }