"""
Author: Jichao Li <cfdljc@gmail.com>

Some functions are copied from SMT (https://github.com/SMTorg/smt)

TODO:
- change corr and poly to FunctionType
"""

from __future__ import division

import numpy as np
from smt.utils.options_dictionary import OptionsDictionary
from smt.utils.pairwise import manhattan_distances
from smt.utils.kriging_utils import componentwise_distance_PLS, ge_compute_pls
from smt.utils.kriging_utils import abs_exp, squar_exp, constant, linear, quadratic

class localGEKPLS(object):
    """
    Reconstruct a GEKPLS model.
    It can provide predictions of drag/lift/picting moment coefficients with respect to inputs.
    Also, gradient of predictions to inputs are provided.
    In addition, it provides the probobility of given inputs belonging to this local model,
    which is further used to compute the mixture propotion in the Mixture of Experts.
    """    
    
    def __init__(self, **kwargs):
        """

        Examples
        --------
        >>> from methods import localGEKPLS
        >>> local1 = localGEKPLS(para_file='data/local1.npy')
        """
        self.options = OptionsDictionary()
        declare = self.options.declare
        declare('para_file', values=None, types=str,
        desc='Directory for loading / saving cached data; None means do not save or load')

        self.options.update(kwargs)
        self._readPara()
        
    def _readPara(self):
        """
        Read parameters from self.options['para_file']
        """
        ## Load the dict using Numpy
        paradict = np.load(self.options['para_file']).item()
        
        ## Copy items outside
        self.X_mean         = paradict['X_mean']
        self.X_std          = paradict['X_std']
        self.X_norma        = paradict['X_norma']
        self.y_mean         = paradict['y_mean']
        self.y_std          = paradict['y_std']
        self.optimal_theta  = paradict['optimal_theta']
        self.nt             = paradict['nt']
        self.optimal_par    = paradict['optimal_par']
        self.corr           = paradict['corr']
        self.n_comp         = paradict['n_comp']
        self.coeff_pls      = paradict['coeff_pls']
        self.poly           = paradict['poly']
        
        del paradict
        
    def _componentwise_distance(self,dx,opt=0):

        d = componentwise_distance_PLS(dx,self.corr,self.n_comp,self.coeff_pls)
        return d

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std

        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=
                                 False)
        d = self._componentwise_distance(dx)

        # Compute the correlation function
        if self.corr == 'abs_exp':
            r = abs_exp(self.optimal_theta, d).reshape(n_eval,self.nt)
        else:
            r = squar_exp(self.optimal_theta, d).reshape(n_eval,self.nt)

        y = np.zeros(n_eval)

        # Compute the regression function
        if self.poly == 'constant':
            f = constant(x)
        elif  self.poly == 'linear':   
            f = linear(x)
        else:
            f = quadratic(x)
        
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par['beta']) + np.dot(r,
                    self.optimal_par['gamma'])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()

        return y        

