import sys
import numpy as np
from methods.MoE import MoE
import matplotlib.pyplot as plt
from scipy import linalg

checktxt=np.loadtxt('Validating.txt')
dim = int((checktxt.shape[1] - 3)/4)
ntest = checktxt.shape[0]
Xtest = checktxt[:,:dim].copy()

nlocal = 14

Cl = MoE(func='Cl',nlocal=nlocal)
ycl = Cl.predict(Xtest)
print 'KRG,  err: '+str(linalg.norm(ycl-checktxt[:,dim])/linalg.norm(checktxt[:,dim])*100)


Cd = MoE(func='Cd',nlocal=nlocal)
ycd = Cd.predict(Xtest)
print 'KRG,  err: '+str(linalg.norm(ycd-checktxt[:,dim+1])/linalg.norm(checktxt[:,dim+1])*100)


Cm = MoE(func='Cm',nlocal=nlocal)    
ycm = Cm.predict(Xtest)
print 'KRG,  err: '+str(linalg.norm(ycm-checktxt[:,dim+2])/linalg.norm(checktxt[:,dim+2])*100)


#cl
plt.plot(checktxt[:,dim],checktxt[:,dim],'-')
plt.plot(checktxt[:,dim],ycl,'o')
plt.show()

#cd
plt.plot(checktxt[:,dim+1],checktxt[:,dim+1],'-')
plt.plot(checktxt[:,dim+1],ycd,'o')
plt.show()

#cm
plt.plot(checktxt[:,dim+2],checktxt[:,dim+2],'-')
plt.plot(checktxt[:,dim+2],ycm,'o')
plt.show()

