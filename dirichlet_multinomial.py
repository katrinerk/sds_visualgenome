import numpy as np
from scipy.special import gamma, factorial

def logddirmult(xvec, alpha):
    # k= number of outcomes
    k = len(xvec)
    # n = number of observations
    n = np.sum(xvec)
    
    if isinstance(alpha, np.ndarray): alphavec = alpha
    elif isinstance(alpha, list) : alphavec = np.array(alpha)
    else: alphavec =  np.ones(k) * alpha
        
    sum_alpha = np.sum(alphavec)

    # p(xvec | n, alphavec) = Gamma(alpha0) Gamma(n+1) / Gamma(n+alpha0) *
    #   prod_i=1^k Gamma(xi + alpha_i) / [ Gamma(alpha_i) Gamma(xi + 1) ]
    # for alpha0 = sum(alpha)
    # so
    #
    # logp(xvec | n, alphavec) = log Gamma(alpha0) + log Gamma(n+1) - log Gamma( n+alpha0) +
    #   sum_i=1^k log Gamma(xi + alpha_i) - log Gamma(alpha_i) - log Gamma(xi + 1)
    d = np.log(gamma(sum_alpha))
    d += np.log(factorial(n))
    d -= np.log(gamma(n + sum_alpha))

    for i in range(k):
        d += np.log(gamma(xvec[i] + alphavec[i]))
        d -= np.log(gamma(alphavec[i]))
        d -= np.log(factorial(xvec[i]))

    return d

                    
