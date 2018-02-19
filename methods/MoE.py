"""
Author: Jichao Li <cfdljc@gmail.com>

Some functions are copied from SMT (https://github.com/SMTorg/smt)

TODO:
- change corr and poly to FunctionType
"""

from __future__ import division
import sys
import numpy as np
from smt.utils.options_dictionary import OptionsDictionary
from classification import eval_gaussianClassifier
from localmodel import localGEKPLS

class MoE(object):
    """
    Construct a global model for Cl/Cd/Cm
    """        
    def __init__(self, **kwargs):
        """

        Examples
        --------
        >>> from methods import localGEKPLS
        >>> clmodel = MoE(func='cl',nlocal=20)
        """
        self.options = OptionsDictionary()
        declare = self.options.declare
        declare('func', values=('Cl', 'Cd', 'Cm'), types=str,desc='Which model to construct')
        declare('nlocal', values=None, types=int,desc='How many local models to use')

        self.options.update(kwargs)
        self.models = []
        self.posteriors = []

        self._setup()
        
    def _setup(self):
        """
        1. Reload local models 
        2. Reload GMM dict
        """
        nlocal = self.options['nlocal']
        funcname = self.options['func']
        
        for i in xrange(nlocal):
            thispath = './data/'+funcname+str(i)+'.npy'
            thislocal = localGEKPLS(para_file=thispath)
            self.models.append(thislocal)
        
        self.GGMinfo = np.load('./data/GMM_'+funcname+'.npy').item()
        
    def _posteriors(self,Xinput):
        """
        Provide posteriors of prediction points Xinput
        
        Parameters
        ---------
        Xinput : np.ndarray [nevals, dim]
               Evaluation point input variable values

        Returns
        -------
        posteriors : np.ndarray [nevals, ncluster]
               posteriors of each points on each local model
        """    
        nevals = Xinput.shape[0]
        dim = Xinput.shape[1]
        nc = int((dim - 2)/2)
        
        #xdata = np.zeros((nevals,4))
        xdata = np.zeros((nevals,2))
        xdata[:,0] = Xinput[:,   nc].copy()    
        xdata[:,1] = Xinput[:,    0].copy()    
        #xdata[:,2] = Xinput[:,dim-2].copy()
        #xdata[:,3] = Xinput[:,dim-1].copy()
        
        indevalsClusters, posteriors, classList, clusterCount = eval_gaussianClassifier(xdata, self.GGMinfo['pi'], self.GGMinfo['mu'], self.GGMinfo['Sigma'], weight=3.0)    
        
        return posteriors

    def predict(self,Xinput):
        """
        Provide predictions
        
        Parameters
        ---------
        Xinput : np.ndarray [nevals, dim]
               Evaluation point input variable values

        Returns
        -------
        Yhat   : np.ndarray [nevals]
               predictions of each points 
        """  
        nevals = Xinput.shape[0]
        nlocal = self.options['nlocal']
        
        posteriors = self._posteriors(Xinput)
        
        # Gather predictions of all local models
        localys = []
        for ilocal in xrange(nlocal):
            thisy = self.models[ilocal]._predict_values(Xinput)
            localys.append(thisy.copy())

        # for the weighted average prediction
        Yhat = np.zeros(nevals)
        for ip in xrange(nevals):
            for ilocal in xrange(nlocal):
                Yhat[ip] += posteriors[ip,ilocal]*localys[ilocal][ip]
        
        return Yhat
        
        
      
