import sys
from preprocess.classification import eval_gaussianClassifier
from methods.localmodel import localGEKPLS


class Airfoil(object):
    
    def __init__(self,**kwargs):
        '''
        Initing funciton will do the following things:
            1. Read existing airfoil data
            2. Read modes
            3. Load local models
        '''
        
        self.inputdata = {}

        # 1. read existing data
        f = open('./data/existing/db_name.txt','r')
        self.exist = {}
        
        self.exist['namelist']=[]
        while True:
            filename=f.readline()
            if filename:
                tempname = filename.rstrip()
                if len(tempname) <30:
                    myname = tempname+' '*(30-len(tempname))
                else:
                    myname = tempname   
                self.exist['namelist'].append(myname.upper())
            else:
                break
        f.close()

        existingdata=np.loadtxt('./data/existing/db_airfoil.txt')
        self.exist['xcor'] = existingdata[0 ,:].copy()
        self.exist['ycors']= existingdata[1:,:].copy()
        
        #2. read modes
        modes_subsonic  = np.loadtxt('./data/modes.subsonic')
        modes_transonic = np.loadtxt('./data/modes.transonic')
        self.xcors = {}
        self.modes = {}
        self.nc={}
        self.nt={}
        self.xcors['subsonic']  = modes_subsonic[0,:].copy()
        self.xcors['transonic'] = modes_transonic[0,:].copy()
        
        self.modes['subsonic']  = modes_subsonic[1:,:].copy()
        self.modes['transonic'] = modes_transonic[1:,:].copy()
        
        self.nc['subsonic']  = int((modes_subsonic.shape[0] - 1)/2)
        self.nc['transonic'] = int((modes_transonic.shape[0] - 1)/2)
        self.nt['subsonic']  = int((modes_subsonic.shape[0] - 1)/2)
        self.nt['transonic'] = int((modes_transonic.shape[0] - 1)/2)

    




def predict_values(self, x):
    
    # ------------------------------------------------------------------------------
    # Compute the posterior probability of each evaluation point
    # ------------------------------------------------------------------------------
    nevals = x.shape[0] 
    xdata = np.zeros((nevals,4))
    xdata[:,0] = x[:,self.nc].copy()    
    xdata[:,1] = x[:,      0].copy()    
    xdata[:,2] = x[:,self.dim-2].copy()
    xdata[:,3] = x[:,self.dim-1].copy()

    indevalsClusters, posteriors, classList, clusterCount = functions.eval_gaussianClassifier(xdata, self.gc_pi, self.gc_mu, self.gc_Sigma, weight=3.0)
    
    for i in xrange(nevals):
        print np.sum(posteriors[i,:])*100.0,':'
        for j in xrange(self.ncluster):
            if posteriors[i,j] > 0.01:
                print j,posteriors[i,j]*100.0



