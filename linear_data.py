import numpy as np
from scipy.sparse import random as random_sparse
from scipy import stats
def generate_linear_data(N, n, n_rel=-1, data_rel_var=1.0 ,data_irrel_var=1.0, noise_var=0.5, correlated=False, standardized=True):
    '''
    N: dimension of data (excluding intercept)
    n_rel: Number of relevant predictor variables
    noise_var: Variance of noise
    standardized: Returns standardized model if true
    '''
    noise = np.random.randn(N)*noise_var
    if n_rel==-1:
        n_rel=n
    # relevant x
    X_rel = np.random.randn(N,n_rel)*data_rel_var
    ones = np.ones(N)
    X = np.c_[ones, X_rel]
    beta = np.random.randn(n_rel+1) ## intercept is included
    if n-n_rel >0:
        X_irrrel = np.random.randn(N,n-n_rel)*data_irrel_var
        X=np.c_[X,X_irrrel]
        beta_irrel = np.zeros(n-n_rel)
        beta=np.concatenate((beta,beta_irrel), axis=0)
    if correlated:
        rvs = stats.poisson(30, loc=10).rvs
        rs = random_sparse(n, n, density=0.1, data_rvs=rvs)
        # rs = np.random.binomial(1,0.1,size=(n,n)).astype('int')
        rs+=np.eye(n).astype('int')
        print(rs)
        X[:,1:] = X[:, 1:].dot(rs)
    y = X.dot(beta)+noise #+ 0.005*(X**2).dot(beta)
    # y = (X**2).dot(beta)+noise
    
    if not standardized:
        return [X,y,beta]
    
    dataset=np.c_[X,y][:, 1:] # remove intercept term
    mean = dataset.mean(axis=0)
    sq_diff = (dataset-mean)**2
    sq_diff = sq_diff.sum(axis=0)/(N-1)
    std = np.sqrt(sq_diff)
    
    beta_reg=beta[1:]*std[:-1]/std[-1]
    dataset = (dataset-mean)/(std*np.sqrt(N-1))
    return [dataset[:,:-1], dataset[:, -1], beta_reg]