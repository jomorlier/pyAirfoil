"""
Author: Jichao Li <cfdljc@gmail.com>

Classification of labelled data
Some functions are copied from previous scripts by Dr. Rhea Liem
"""
import numpy as np


# -----------------------------------------------------------------------------
# Argument for the logistic function (quadratic)
# -----------------------------------------------------------------------------

def computeaquadratic(pi, mu, Sigma, x):
    # the quadratic term
    qterm = np.linalg.solve(Sigma, x)
    quadTerm = -1.0/2.0 * np.dot(x, qterm)

    # vector b: for the linear term
    b = np.linalg.solve(Sigma, mu)
    linTerm = np.dot(b,x) 

    # vector c: for the constant term
    cterm1 = -1.0/2.0*np.dot(mu, np.linalg.solve(Sigma, mu))
    cterm2 = -1.0/2.0*np.log(np.linalg.det(Sigma))
    cterm3 =  np.log(pi)

    constTerm = cterm1 + cterm2 + cterm3

    a = quadTerm + linTerm + constTerm

    return a

# -----------------------------------------------------------------------------
# Normal PDF
# -----------------------------------------------------------------------------

def normalPDF(x, mu, s2):
    pdf = np.sqrt(1.0/2*np.pi)*np.sqrt(1.0/s2)*np.exp(-((x-mu)**2)/(2*s2))

    return pdf

    
# -----------------------------------------------------------------------------
# Log likelihood for the mixture of Gaussians (MOG)
# -----------------------------------------------------------------------------

def mog_logLikelihood(x, piVec, muVec, sigma2Vec):
    # number of data
    N = len(x)
    nK = len(piVec)

    # log likelihood 
    L = 0.0

    for i in xrange(N):
        # compute the likelihood of the i-th data 
        xi = x[i,0]
        tmp = 0.0
        for k in xrange(nK):
            pik = piVec[k]
            muk = muVec[k]
            s2k = sigma2Vec[k]

            pdf = normalPDF(xi, muk, s2k)
            tmp += pik * pdf

        if tmp < 1e-5:
            tmp = 1e-5
        L += np.log(tmp)

    return L

# ------------------------------------------------------------------------------
# SUPERVISED LEARNING ALGORITHM
# To provide a classification in the x-space
# Use the classified training samples (from the unsupervised learning algorithm
# results) as the training data.
# Training data: {xn, tn}, n = 1, ..., N
# ------------------------------------------------------------------------------

def gaussianclassifier(x,t,regularize=False, regConstant=0.01):
    # x: data points
    # t: classification inputs 

    # problem dimension (number of variables)
    Ndv = x.shape[1]

    # number of data
    N = len(t)

    # number of clusters
    nclusters = len(np.unique(t))

    # number of data within each cluster 
    nmembers = np.zeros(nclusters, dtype=int) 

    # clusters' prior probabilities
    pi = np.zeros(nclusters)

    # indices of data for each cluster
    indmembers = []
    xmembers = []
    # initialize
    for j in xrange(nclusters):
        indmembers.append([])
        xmembers.append([])

    for j in xrange(nclusters):
        indmembers[j] = [i for (i,val) in enumerate(t) if val == j]
        xmembers[j] = x[indmembers[j],:].copy()
        nmembers[j] = len(indmembers[j])

        pi[j] = np.float(nmembers[j])/np.float(N)

    # compute mean of each cluster
    mu = np.zeros((nclusters, Ndv))

    for j in xrange(nclusters):
        for d in xrange(Ndv):
            mu[j,d] = np.sum(xmembers[j][:,d])
        mu[j,:] = mu[j,:]/nmembers[j]

    # compute the class covariance matrix
    Sigma = []
    # initialize
    for j in xrange(nclusters):
        Sigma.append([])

    for j in xrange(nclusters):
        Sigma[j] = np.zeros((Ndv, Ndv)) 

        for n in xrange(nmembers[j]):
            tmp_Sigma = np.zeros((Ndv, Ndv)) 

            vec = xmembers[j][n,:] - mu[j,:]
            for p in xrange(Ndv):
                for r in xrange(Ndv): 
                    tmp_Sigma[p,r] = vec[p]*vec[r]

            Sigma[j] += tmp_Sigma 

        Sigma[j] = 1.0/np.float(nmembers[j]) * Sigma[j]

    if regularize:
        reg = np.zeros((Ndv, Ndv))

        for d in xrange(Ndv):
            reg[d,d] = regConstant 

        for j in xrange(nclusters):
            Sigma[j] += reg

    return pi, mu, Sigma

# -----------------------------------------------------------------------------
# Evaluate the posteriors of the Gaussian classifier 
# When we already have the pi, mu, and Sigma
# -----------------------------------------------------------------------------

def eval_gaussianClassifier(xevals, pi, mu, Sigma, weight=1.0, bias=0.0):
    Nevals = xevals.shape[0]
    Ndv = xevals.shape[1]
    nclusters = len(pi)

    posteriors = np.zeros((Nevals, nclusters))
    classList = np.zeros(Nevals, dtype=int)
    clusterCount = np.zeros(nclusters, dtype=int)

    for n in xrange(Nevals):
        x = xevals[n,:]

        post = np.zeros(nclusters)
        for j in xrange(nclusters):
            # compute the argument of the logistic function
            post[j] = computeaquadratic(pi[j], mu[j,:], Sigma[j], x)
            post[j] = np.exp(weight*post[j]+bias)
            
        # normalize the posterior probability
        post = post/np.sum(post)
        
        posteriors[n,:] = post
        
        # sort the posterior probability from lowest to highest
        indSort = np.argsort(post)
        
        # find the cluster index with the highest posterior probability 
        indClass = indSort[-1]

        classList[n] = indClass

        # add the counter 
        clusterCount[indClass] += 1

    indevalsClusters = []
    # initialize 
    for j in xrange(nclusters):
        indevalsClusters.append(np.zeros(clusterCount[j], dtype=int))

    for j in xrange(nclusters):
        indevalsClusters[j] = [i for (i,val) in enumerate(classList) if val == j]
    
    return indevalsClusters, posteriors, classList, clusterCount
 
