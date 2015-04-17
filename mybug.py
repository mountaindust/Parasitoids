# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
I am a bug

"""
import numpy as np
class mybug:
    def __init__(self,y=None):
        if y is None:
            self.x = np.array([0,0])
        else: 
            self.x = np.array(y)
        self.history = [self.x]
            
    def update_position(self):
        mean=np.array([0,0])
        y = np.random.multivariate_normal(mean,np.eye(2))
        self.x = self.x + y
        self.history.append(self.x)