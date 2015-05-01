#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swarm file

Created on Fri Apr 17 10:35:55 2015
"""

import mybug
from matplotlib import pyplot as plt
import numpy as np
class myswarm:
    
    def __init__(self,numbugs=None):
        if numbugs is None:
            numbugs = 1
        self.swarm = []
        for ii in xrange(numbugs):
            self.swarm.append(mybug.mybug())
            
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
        clrs = np.linspace(0,1,len(self.swarm))
        plt.hold(True)
        plt.scatter(poslist[:,0],poslist[:,1],c=clrs)
        for n,bug_hist in enumerate(swhistory):
            bug_hist = np.array(bug_hist)
            plt.plot(bug_hist[:,0],bug_hist[:,1])
        plt.show()