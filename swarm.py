#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swarm class file, for simulating and plotting many bug objects at once

Created on Fri Apr 17 10:35:55 2015
"""

import mybug
from matplotlib import pyplot as plt
import matplotlib.cm as colormp
import numpy as np
class myswarm:
    
    #initialize swarm with a given number of bugs (default 1)
    def __init__(self,numbugs=None):
        if numbugs is None:
            numbugs = 1
        self.swarm = []
        #bug objects go in a list
        for ii in xrange(numbugs):
            self.swarm.append(mybug.mybug())
    
    #method for updating the position of the entire swarm.
    #spatially uniform wind vector can be given
    def update_swarm(self,n=1,wind_array=None):
        if wind_array is None:
            wind_array = np.zeros((n,2))
        else:
            assert type(wind_array is np.ndarray), \
            'wind_array must be a numpy ndarray of shape n x 2'
            assert wind_array.shape == (n,2), \
            'wind_array must have shape n x 2'
        for ii in xrange(n):
            for bug in self.swarm:
                bug.update_position(wind_array[ii,:])
                
    def plot_swarm(self):
        poslist = []
        swhistory = []
        for bug in self.swarm:
            poslist.append(bug.x)
            swhistory.append(bug.history)
        poslist = np.array(poslist)
        #color our bugs!
        cm = colormp.get_cmap() #get standard colormap
        clrs = np.linspace(0,1,len(self.swarm))
        plt.hold(True)
        plt.scatter(poslist[:,0],poslist[:,1],c=clrs)
        for n,bug_hist in enumerate(swhistory):
            bug_hist = np.array(bug_hist)
            plt.plot(bug_hist[:,0],bug_hist[:,1],c=cm(clrs[n]))
        plt.axis('scaled') #force x and y axis to be the same units
        plt.show()