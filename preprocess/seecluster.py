import functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from sklearn.cluster import KMeans

clusterdict = np.load('clusterCd.npy').item()
subset='M1A0'
nc = len(clusterdict[subset])

colors=[]
cmap = matplotlib.cm.get_cmap('viridis')
for i in xrange(nc):
    colors.append(cmap(1.0*i/(nc-1.0)))


fig = plt.figure(figsize=(10.0,4.5))
gs = gridspec.GridSpec(1,1)#,width_ratios=[1.0,0.1])
ax = fig.add_subplot(gs[0,0])
ax.set_aspect('equal', 'datalim')
ax.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=True,      # ticks along the bottom edge are off
                labelleft=True,
                labelright=False,
                labelbottom=True,
                right=False,         # ticks along the top edge are off
                bottom=True,
                top=False,
                colors='#2c3e50',
                labelsize=13.0,
                width=0.2
                )
ax.set_xlabel('The first thickness mode',fontsize=16.0,color='#2c3e50')
ax.set_ylabel('The first camber mode',fontsize=16.0,color='#2c3e50')
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color('#bdc3c7')
ax.spines['left'].set_color('#bdc3c7')
ax.spines['left'].set_linewidth(0.2)
ax.spines['bottom'].set_linewidth(0.2)            
for i in xrange(nc):
    data = clusterdict[subset]['cluster'+str(i)]
    ax.scatter(data[:,7],data[:,0],marker='.',s=1.0,c=colors[i])
    
fig.savefig('plot.eps',dpi=1000, bbox_inches="tight")                


