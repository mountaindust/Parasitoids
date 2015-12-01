#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot stuff

Created on Sat Mar 07 20:35:29 2015

@author: Christopher Strickland
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pymc as pm

#norm = stats.norm
#x = np.linspace(-10, 10, 1000)
#mu = 0
#sig2 = 1
#plt.plot(x, norm.pdf(x,mu,sig2))
#plt.show()

x = np.linspace(0.01, 5, 1000)
gamln = np.zeros(1000)
for ii in range(len(x)):
    gamln[ii] = pm.gamma_like(x[ii],2,1)
plt.plot(x, np.exp(gamln))
#plt.hold(True)
#plt.plot(x, gam.pdf(x,4,0,0.25))
plt.show()