#! /usr/bin/env python3

'''
This module is for plotting the posterior distributions from Bayes_Run.py

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

import warnings
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['image.cmap'] = 'viridis'
cmap = cm.get_cmap('viridis')

database_name = 'mcmcdb.h5'

db = pm.database.hdf5.load(database_name)

plt.ion()



def plot_traces(db=db):
    '''Plot the traces of the unknown variables to check for convergence'''
    lw=1 #line width
    
    plt.figure()
    plt.hold(True)
    
    plt.subplot(411)
    plt.title("Traces of unknown model parameters")
    # f: a_1,a_2
    plt.plot(db.trace("a_1",chain=None)[:], label=r"trace of $f:a_1$",
             c=cmap(0.3), lw=lw)
    plt.plot(db.trace("a_2",chain=None)[:], label=r"trace of $f:a_2$",
             c=cmap(0.7), lw=lw)
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(0.7)
    
    # f: b_1,b_2, g: a_w,b_w
    plt.subplot(412)
    plt.plot(db.trace("f_b1",chain=None)[:], label=r"trace of $f:b_1$",
             c=cmap(0.3), lw=lw)
    plt.plot(db.trace("f_b2",chain=None)[:], label=r"trace of $f:b_2$",
             c=cmap(0.7), lw=lw)
    plt.plot(db.trace("a_w",chain=None)[:], label=r"trace of $g:a_w$",
             c=cmap(0.1), lw=lw)
    plt.plot(db.trace("b_w",chain=None)[:], label=r"trace of $g:b_w$",
             c=cmap(0.9), lw=lw)
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(0.7)
    
    # sig_x,sig_y,sig_x_l,sig_y_l
    plt.subplot(413)
    plt.plot(db.trace("sig_x",chain=None)[:], label=r"trace of $\sigma_x$",
             c=cmap(0.01), lw=lw)
    plt.plot(db.trace("sig_y",chain=None)[:], label=r"trace of $\sigma_y$",
             c=cmap(0.99), lw=lw)
    plt.plot(db.trace("sig_xl",chain=None)[:], label=r"trace of local $\sigma_x$",
             c=cmap(0.25), lw=lw)
    plt.plot(db.trace("sig_yl",chain=None)[:], label=r"trace of local $\sigma_y$",
             c=cmap(0.75), lw=lw)
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(0.7)
    
    # corr,corr_l,lam
    plt.subplot(414)
    plt.plot(db.trace("rho",chain=None)[:], label=r"trace of $\rho$",
             c=cmap(0.01), lw=lw)
    plt.plot(db.trace("rho_l",chain=None)[:], label=r"trace of local $\rho$",
             c=cmap(0.25), lw=lw)
    plt.plot(db.trace("lam",chain=None)[:], label=r"trace of $\lambda$",
             c=cmap(0.5), lw=lw)
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(0.7)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)
    
    plt.figure()
    plt.hold(True)
    
    plt.subplot(211)
    plt.title("Traces of unknown Bayesian model parameters")
    # xi, em_obs_prob, grid_obs_prob
    plt.plot(db.trace("xi",chain=None)[:], label=r"trace of $\xi$",
             c=cmap(0.01), lw=lw)
    plt.plot(db.trace("em_obs_prob",chain=None)[:], 
             label=r"trace of emerg obs prob", c=cmap(0.5), lw=lw)
    plt.plot(db.trace("grid_obs_prob",chain=None)[:], 
             label=r"trace of grid obs prob", c=cmap(0.99), lw=lw)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    # sent_obs_probs
    plt.subplot(212)
    n = 0
    for name in db.trace_names[0]:
        if name[:13] == 'sent_obs_prob':
            id = name[-1]
            plt.plot(db.trace(name,chain=None)[:], 
                     label="trace of obs prob field {}".format(id),
                     c=cmap(.10+n*.16), lw=lw)
            n += 1
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    plt.show()



def plot_f_g(db=db,start=0):
    '''Plot the posterior distributions for the f and g model functions.
    
    Arguments:
        db: database object
        start: where to begin in the trace (with all chains taken together)
    '''
    plt.figure()
    ax = plt.subplot(311) # f-a_1, f-a_2
    # Get color cycler
    color_cycle = ax._get_lines.color_cycle
    plt.title(r"Posterior distributions of $f$ and $g$ model functions")
    plt.xlim(0,24)
    plt.hold(True)
    plt.hist(db.trace("a_1",chain=None)[start:], histtype='stepfilled', bins=27,
             alpha=0.85, label=r"posterior of position param $a_1$",
             color=next(color_cycle), normed=True)
    plt.hist(db.trace("a_2",chain=None)[start:], histtype='stepfilled', bins=27,
             alpha=0.85, label=r"posterior of position param $a_2$",
             color=next(color_cycle), normed=True)
    plt.hold(False)
    plt.legend(loc="upper center")

    ax = plt.subplot(312) #f-b_1, f-b_2
    plt.xlim(1,15)
    plt.hold(True)
    plt.hist(db.trace("f_b1",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $b_1$",
             color=next(color_cycle), normed=True)
    plt.hist(db.trace("f_b2",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $b_2$",
             color=next(color_cycle), normed=True)
    plt.hold(False)
    plt.legend(loc="upper right")

    ax = plt.subplot(313) #g-a_w, g-b_w
    plt.xlim(0,15)
    # Get new color cycler
    color_cycle = ax._get_lines.color_cycle
    plt.hold(True)
    plt.hist(db.trace("a_w",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of position param $a_w$",
             color=next(color_cycle), normed=True)
    unused = next(color_cycle)
    plt.hist(db.trace("b_w",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $b_w$",
             color=next(color_cycle), normed=True)
    plt.hold(False)
    plt.legend(loc="upper right")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)