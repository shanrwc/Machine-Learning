#!/usr/bin/python
import numpy
import matplotlib.pyplot as plt

#plot types
plt.figure(figsize=(8,8),dpi=80,facecolor='w')
plt.plot(xs,other_ys,color='r',label='Plot')

#bins for histograms can be an array of boundaries or just a number of bins
#histtype can be step, stepfilled, bar, 
#Other options: cumulative=True or -1, normed=1, stacked=True
plt.hist(xs,<num of bins>,histtype='step',color='r')
plt.scatter(xs,ys,color='b',label='Name',)

#colors: r-red, g-green,b-blue,c-cyan,m-magenta,y-yellow,k-black,w-white
#line styles: -solid, --dashed, :dotted, -.dot-dashed,.points,o filled circles,^filled triangles

plt.title("My Plot")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.axis([x_min,x_max,y_min,y_max])#you can also use plt.xlim() and plt.ylim()
plt.legend()
#can take a loc option such as 'upper right'

#add a grid
plt.grid(True)

plt.show()
plt.savefig(plotname)
#format of output figure is determined by the extension string
plt.clf()
#This clears the figure so your figures don't overlap

##################################################################

#The following is from the Udacity machine learning course and was used
#to plot a decision boundary
import numpy as np

h = 0.01
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.pcolormesh(xx,yy,Z,cmap=pl.cm.seismic)
