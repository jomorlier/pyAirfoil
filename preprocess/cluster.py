"""
Author: Jichao Li <cfdljc@gmail.com>

Split a large dataset into clusters.
It contains two steps:
- Hard split dataset into sub-ones. Suitable for Mach and Alpha
- Split each sub-one into clusters using K-means algorithm. The first camber and thickness mode plus the function
"""
import numpy as np
from sklearn.cluster import KMeans

def split(filename, nMa, nAoA, ncluster):
    """
    
    Parameters
    ----------
    filename: String.
              - The filename where the dataset is stored.
    ncluster: Int
              - Number of clusters in each sub-dataset
    nMa     : Int
              - Number of intervals to split Mach
    nAoA    : Int
              - Number of intervals to split AoA
    
    Returns
    ----------
    Three files are written out: 'clusterCl.npy','clusterCd.npy, and 'clusterCm.npy' 
    
    """    
    # Read dataset. Each row contains the input and cl, cd, cm, and gradient of cl,cd,cm to inputs
    X = np.loadtxt(filename)
    
    # The dimension of input
    dim = int((X.shape[1] - 3)/4)
    ns = X.shape[0]
    # Number of camber modes
    nc  = int((dim-2)/2)

    dictCl = {}
    dictCd = {}
    dictCm = {}
    
    # Hard split for Mach and AoA
    minMach = np.min(X[:,dim-2]) - (np.max(X[:,dim-2]) - np.min(X[:,dim-2]))*0.001     
    maxMach = np.max(X[:,dim-2]) + (np.max(X[:,dim-2]) - np.min(X[:,dim-2]))*0.001             
    minAoA  = np.min(X[:,dim-1]) - (np.max(X[:,dim-1]) - np.min(X[:,dim-1]))*0.001             
    maxAoA  = np.max(X[:,dim-1]) + (np.max(X[:,dim-1]) - np.min(X[:,dim-1]))*0.001             
    
    # Margins in hard split. For slight overlap
    marginMach = (maxMach - minMach)*0.015
    marginAoA  = (maxMach - minMach)*0.015
    
    icluster = 0
    
    # global labels
    cllabel = -1*np.ones(ns, dtype=int)
    cdlabel = -1*np.ones(ns, dtype=int)
    cmlabel = -1*np.ones(ns, dtype=int)
    
    dictCl['subset'] = []
    dictCd['subset'] = []
    dictCm['subset'] = []
    
    for im in xrange(nMa):
        # define Mach bounds of this sub-set
        mach_lower = minMach + (maxMach - minMach)*im/nMa - marginMach
        mach_upper = minMach + (maxMach - minMach)*(im+1.0)/nMa + marginMach
        for ia in xrange(nAoA):
            # define AoA bounds of this sub-set
            AoA_lower = minAoA + (maxAoA - minAoA)*ia/nAoA - marginAoA
            AoA_upper = minAoA + (maxAoA - minAoA)*(ia+1.0)/nAoA + marginAoA
            
            #  check if a point is strictly belong to this sub-set
            def strictbelong(myM,myAoA):
                #return True
                if myM > mach_lower+marginMach and myM <= mach_upper-marginMach and myAoA > AoA_lower+marginAoA and myAoA <= AoA_upper-marginAoA:
                    return True
                else:
                    return False
            
            # select which points are in this sub-set    
            subindex=[]
            for i in xrange(ns):
                if X[i,dim-2] > mach_lower and X[i,dim-2] <= mach_upper and X[i,dim-1] > AoA_lower and X[i,dim-1] <= AoA_upper:
                    subindex.append(i)
            subns = len(subindex)
            
            # Restore their first thickness, first camber, and cl/cd/cm
            subcl = np.zeros((subns,3))
            subcd = np.zeros((subns,3))
            subcm = np.zeros((subns,3))
            for i in xrange(subns):
                subcl[i,0] = X[subindex[i],nc]
                subcl[i,1] = X[subindex[i], 0] 
                subcl[i,2] = X[subindex[i],dim] 

                subcd[i,0] = X[subindex[i],nc]
                subcd[i,1] = X[subindex[i], 0] 
                subcd[i,2] = X[subindex[i],dim+1] 

                subcm[i,0] = X[subindex[i],nc]
                subcm[i,1] = X[subindex[i], 0] 
                subcm[i,2] = X[subindex[i],dim+2] 

            # Scale cl/cd/cm to same magnitude with Camber to obtain more reasonable clusters
            scalecl = 0.5*np.linalg.norm(subcl[:,1])/np.linalg.norm(subcl[:,2])
            subcl[:,2] = scalecl*subcl[:,2]

            scalecd = 0.5*np.linalg.norm(subcd[:,1])/np.linalg.norm(subcd[:,2])
            subcd[:,2] = scalecd*subcd[:,2]

            scalecm = 0.5*np.linalg.norm(subcm[:,1])/np.linalg.norm(subcm[:,2])
            subcm[:,2] = scalecm*subcm[:,2]
            
            # K-Means clustering
            kmeansCl = KMeans(n_clusters=ncluster,random_state=0,n_init=10).fit(subcl)
            kmeansCd = KMeans(n_clusters=ncluster,random_state=0,n_init=10).fit(subcd)
            kmeansCm = KMeans(n_clusters=ncluster,random_state=0,n_init=10).fit(subcm)
            
            # Use dicts to store points in each clusters
            Clcluster = {}
            Cdcluster = {}
            Cmcluster = {}
            
            
            for i in range(ncluster):
                localcl = []
                localcd = []
                localcm = []
                for j in range(subns):
                    if kmeansCl.labels_[j] == i:
                        localcl.append(subindex[j])
                        if strictbelong(X[subindex[j],dim-2],X[subindex[j],dim-1]):
                            cllabel[subindex[j]] = icluster
                    if kmeansCd.labels_[j] == i:
                        localcd.append(subindex[j])
                        if strictbelong(X[subindex[j],dim-2],X[subindex[j],dim-1]):
                            cdlabel[subindex[j]] = icluster
                    if kmeansCm.labels_[j] == i:
                        localcm.append(subindex[j])
                        if strictbelong(X[subindex[j],dim-2],X[subindex[j],dim-1]):
                            cmlabel[subindex[j]] = icluster
                clustercl = np.zeros((len(localcl),dim*2+1))
                clustercd = np.zeros((len(localcd),dim*2+1))
                clustercm = np.zeros((len(localcm),dim*2+1))
                for j in xrange(len(localcl)):
                    clustercl[j,:dim]   = X[localcl[j],:dim           ]
                    clustercl[j, dim]   = X[localcl[j], dim           ]
                    clustercl[j,dim+1:] = X[localcl[j],dim+3:2*dim+3  ]                             
                for j in xrange(len(localcd)):
                    clustercd[j,:dim]   = X[localcd[j],:dim           ]
                    clustercd[j, dim]   = X[localcd[j],dim+1          ]
                    clustercd[j,dim+1:] = X[localcd[j],2*dim+3:3*dim+3] 
                for j in xrange(len(localcm)):
                    clustercm[j,:dim]   = X[localcm[j],:dim           ]
                    clustercm[j, dim]   = X[localcm[j],dim+2          ]
                    clustercm[j,dim+1:] = X[localcm[j],3*dim+3:4*dim+3]                                                                   
                
                Clcluster['cluster'+str(i)] = clustercl
                Cdcluster['cluster'+str(i)] = clustercd
                Cmcluster['cluster'+str(i)] = clustercm
                
                icluster += 1
                del clustercl,clustercd,clustercm
            
            # put local dicts into the global ones
            dictCl['subset'].append(Clcluster)
            dictCd['subset'].append(Cdcluster)
            dictCm['subset'].append(Cmcluster)

    # check if all points are labelled
    clmiss = np.all(cllabel > -1)
    cdmiss = np.all(cdlabel > -1)
    cmmiss = np.all(cmlabel > -1)
    if not(clmiss and cdmiss and cmmiss):
        print 'Some points are not labelled.'
        
    dataX = np.zeros((ns,4))

    dataX[:,0]   = X[:,nc]
    dataX[:,1]   = X[:,0]
    dataX[:,2]   = X[:,dim-2]                             
    dataX[:,2]   = X[:,dim-1]    

    dictCl['dataX']  = dataX
    dictCl['label'] = cllabel

    dictCd['dataX']  = dataX
    dictCd['label'] = cdlabel
    
    dictCm['dataX']  = dataX
    dictCm['label'] = cmlabel
                
    # write three global dict to files
    np.save('clusterCl.npy',dictCl)        
    np.save('clusterCd.npy',dictCd)        
    np.save('clusterCm.npy',dictCm)                
    
    '''
    Local models are slightly overlapped.  
    But GMM need non-repeatable marks.
    '''


split('CFDdata.txt', 2, 2, 8)    
    
