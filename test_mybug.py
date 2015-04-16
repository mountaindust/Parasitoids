# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:39:45 2015

@author: sue
"""

import mybug
import warnings
import numpy as np
testdefaultx = mybug.mybug()
assert(np.array_equal(testdefaultx.x,np.array([0,0])))
testinitial = mybug.mybug([1,2])
assert(np.array_equal(testinitial.x,np.array([1,2])))
for ii in xrange(999):
    testdefaultx.update_position()
pos999 = testdefaultx.x 
testdefaultx.update_position()
pos1000 = testdefaultx.x
if not (np.linalg.norm(pos999)>np.linalg.norm(pos1000-pos999)):
    warnings.warn("your bug is a slowpoke")