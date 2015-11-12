# -*- coding: utf-8 -*-
"""
Test file for swarm.py
"""

import swarm
import warnings
import numpy as np

#test default initialization
default_swarm = swarm.myswarm()
#this should have just one bug
assert(len(default_swarm.swarm) == 1)
default_swarm.update_swarm() #update once
default_swarm.update_swarm(5) #update 5 times

#test wind drift
wind_swarm = swarm.myswarm()
wind_swarm.update_swarm(2,np.array([[44,0],[0,117]]))
#since (44,117,125) is a Pythagorean triple, the norm should be near 125
assert(110 < np.linalg.norm(wind_swarm.swarm[0].x) < 140)
assert(39 < wind_swarm.swarm[0].x[0] < 50)
assert(112 < wind_swarm.swarm[0].x[1] < 122)

#test many bugs
bigger_swarm = swarm.myswarm(5)
bigger_swarm.update_swarm()
bigger_swarm.update_swarm(5)
bigger_swarm.update_swarm(2,np.array([[44,3],[50,117]]))