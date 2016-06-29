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
    # previous versions did not have a hypervariable rho_p, so rho is plotted
    try:
        plt.plot(db.trace("rho",chain=None)[:], label=r"trace of $\rho$",
                c=cmap(0.01), lw=lw)
        plt.plot(db.trace("rho_l",chain=None)[:], label=r"trace of local $\rho$",
                c=cmap(0.25), lw=lw)
    except KeyError:
        plt.plot(db.trace("corr",chain=None)[:], label=r"trace of $\rho$",
                c=cmap(0.01), lw=lw)
        plt.plot(db.trace("corr_l",chain=None)[:], label=r"trace of local $\rho$",
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
    color_cycle = ax._get_lines.prop_cycler
    plt.title(r"Posterior distributions of $f$ and $g$ model functions")
    plt.xlim(0,24)
    plt.hold(True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("a_1",chain=None)[start:], histtype='stepfilled', bins=27,
             alpha=0.85, label=r"posterior of position param $a_1$",
             color=clrdict['color'], normed=True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("a_2",chain=None)[start:], histtype='stepfilled', bins=27,
             alpha=0.85, label=r"posterior of position param $a_2$",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper center")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(312) #f-b_1, f-b_2
    plt.xlim(1,15)
    plt.hold(True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("f_b1",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $b_1$",
             color=clrdict['color'], normed=True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("f_b2",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $b_2$",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(313) #g-a_w, g-b_w
    plt.xlim(0,15)
    # Get new color cycler
    color_cycle = ax._get_lines.prop_cycler
    plt.hold(True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("a_w",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of position param $a_w$",
             color=clrdict['color'], normed=True)
    unused = next(color_cycle)
    clrdict = next(color_cycle)
    plt.hist(db.trace("b_w",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $b_w$",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)
        
        
        
def plot_sprd_vars(db=db,start=0):
    '''Plot posteriors of covariance variables for local/wind diffusion, and for
       flight time.
    
    Arguments:
        db: database object
        start: where to begin in the trace (with all chains taken together)
    '''
    
    plt.figure()
    ax = plt.subplot(411)
    plt.title("Posterior distribs for diffusion covariance & flight time")
    plt.hold(True)
    plt.hist(db.trace("sig_x",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label=r"posterior of wind $\sigma_x$",
             normed=True)
    plt.hist(db.trace("sig_y",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label=r"posterior of wind $\sigma_y$",
             normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    ax = plt.subplot(412)
    plt.hold(True)
    plt.hist(db.trace("sig_xl",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label=r"posterior of local $\sigma_x$",
             normed=True)
    plt.hist(db.trace("sig_yl",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label=r"posterior of local $\sigma_y$",
             normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    ax = plt.subplot(413)
    color_cycle = ax._get_lines.prop_cycler
    # get some new colors
    unused = next(color_cycle)
    unused = next(color_cycle)
    clrdict = next(color_cycle)
    plt.hold(True)
    plt.hist(db.trace("rho",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label=r"posterior of wind $\rho$",
             color=clrdict['color'], normed=True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("rho_l",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label=r"posterior of local $\rho$",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    ax = plt.subplot(414)
    color_cycle = ax._get_lines.prop_cycler
    # get a new color
    for ii in range(3):
        unused = next(color_cycle)
    clrdict = next(color_cycle)
    # this is discrete data. need to bin it correctly
    tr = db.trace("t_dur",chain=None)[start:]
    plt.hold(True)
    plt.hist(tr, bins=np.arange(tr.min(),tr.max()+2,1)-.5,
             histtype='stepfilled',  alpha=0.85, 
             label=r"posterior of avg flight time (min)", 
             color=clrdict['color'], normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)
        
        
        
def plot_sent_obs_probs(db=db,start=0):
    '''Plot posteriors for emergence observation probability in each 
    sentinel field.
    
    Arguments:
        db: database object
        start: where to begin in the trace (with all chains taken together)
    '''
    
    # Get sentinel field info
    N_fields = 0
    field_names = []
    field_ids = []
    for name in db.trace_names[0]:
        if name[:13] == 'sent_obs_prob':
            N_fields += 1
            field_names.append(name)
            field_ids.append(name[-1])
    
    plt.figure()
    for ii in range(N_fields):
        ax = plt.subplot(N_fields,1,ii+1)
        if ii == 0:
            plt.title("Posterior distribs for sentinel field emerg obs probs")
        plt.hist(db.trace(field_names[ii],chain=None)[start:], 
                 histtype='stepfilled', bins=25, alpha=0.85, 
                 label="field {}".format(field_ids[ii]),
                 normed=True)
        leg = plt.legend(loc="upper right")
        leg.get_frame().set_alpha(0.7)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)
        
        
        
def plot_other(db=db,start=0):
    '''Plot posteriors for lambda, xi, grid_obs_prob, em_obs_prob and A_collected
    
    Arguments:
        db: database object
        start: where to begin in the trace (with all chains taken together)
    '''
    
    plt.figure()
    ax = plt.subplot(411)
    plt.title(r"Posteriors for $\lambda$, $\xi$, grid_obs_prob and em_obs_prob")
    plt.hist(db.trace("lam",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior for $\lambda$", normed=True)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    ax = plt.subplot(412)
    plt.hist(db.trace("xi",chain=None)[start:], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior for $\xi$", normed=True)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    ax = plt.subplot(413)
    plt.hold(True)
    plt.hist(db.trace("grid_obs_prob",chain=None)[start:], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior for grid_obs_prob",
             normed=True)
    plt.hist(db.trace("em_obs_prob",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label=r"posterior for em_obs_prob",
             normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    ax = plt.subplot(414)
    plt.hist(db.trace("A_collected",chain=None)[start:], histtype='stepfilled', 
             bins=25, alpha=0.85, label="posterior for A_collected", normed=True)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)