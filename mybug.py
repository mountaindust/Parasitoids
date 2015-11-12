# -*- coding: utf-8 -*-
"""
I am a class file for a single bug

"""
import numpy as np

class mybug:
    #initialize the object
    def __init__(self,y=None):
        if y is None:
            self.x = np.array([0,0])
        else: 
            self.x = np.array(y)
        self.history = [self.x]
    
    #method to update the bug's position
    def update_position(self,drift=None):
        if drift is None:
            drift = np.array([0,0])
        mean=np.array([0,0])
        y = np.random.multivariate_normal(mean,np.eye(2))+drift
        self.x = self.x + y
        self.history.append(self.x)