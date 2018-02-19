"""
Author: Jichao Li <cfdljc@gmail.com>

Train local models
"""
import numpy as np
import sys
import os
from smt.methods import GEKPLS,KRG
import classification

funcs = ['Cl','Cd','Cm']
ncoef = 3
extra_points = 0
delta_x = 5e-4

os.system('rm ../data/*')

for ifunc in xrange(len(funcs)):
    imodel = 0
    datadic = np.load('cluster'+funcs[ifunc]+'.npy').item()

    dataX  = datadic['dataX']
    label = datadic['label']
    
    pi, mu, Sigma = classification.gaussianclassifier(dataX, label)
    GMMdict={}
    GMMdict['pi'] = pi
    GMMdict['mu'] = mu
    GMMdict['Sigma'] = Sigma
    np.save('../data/GMM_'+funcs[ifunc]+'.npy',GMMdict)
    
    for isub in xrange(len(datadic['subset'])):
        
        localset = datadic['subset'][isub]
        ncluster = len(localset) 
        
        for icluster in xrange(ncluster):
            data = localset['cluster'+str(icluster)]
            dim  = int((data.shape[1] - 1)/2)
            clusterlimits = np.zeros((dim,2))
            for i in xrange(dim):
                clusterlimits[i,0] = np.min(data[:,i])
                clusterlimits[i,1] = np.max(data[:,i])
            # Local model
            t1 = GEKPLS(n_comp=ncoef, theta0=[0.1]*ncoef, xlimits=clusterlimits, delta_x=delta_x,extra_points= extra_points)
            t1.set_training_values(data[:,:dim],data[:,dim])
            # Add the gradient information
            for i in xrange(dim):
                t1.set_training_derivatives(data[:,:dim],data[:,dim+1+i].reshape((data.shape[0],1)),kx=i)
            t1.train()
            
            t1._saveModeDict('../data/'+funcs[ifunc]+str(imodel)+'.npy')

            imodel += 1

          
      
      

            
