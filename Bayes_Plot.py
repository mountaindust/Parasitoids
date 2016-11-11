#! /usr/bin/env python3

'''
This module is for plotting the posterior distributions from Bayes_Run.py

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

import sys, os
import warnings
from collections import OrderedDict
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['image.cmap'] = 'viridis'
cmap = cm.get_cmap('Accent')

plt.ion()

if __name__ != "__main__":
    database_name = 'mcmcdb.h5'
    db = pm.database.hdf5.load(database_name)
else:
    db = None

def plot_traces(db=db,path='./diagnostics',format='png'):
    '''Plot the traces of the unknown variables to check for convergence.
    Also compute several convergence methods and print them out.'''
    lw = 1 #line width

    # Specify variables to include in each figure and subplot
    #   Each sublist is a figure. Each OrderedDict is a subplot with the
    #   key as the trace name and the val as the LaTeX string name.
    var_names = []
    var_names.append([OrderedDict([('f_a1', r'$f:a_1$'), ('f_a2', r'$f:a_2$')])])
    var_names[0].append(OrderedDict([('f_b1', r'$f:b_1$'), ('f_b2', r'$f:b_2$'),
                                     ('g_aw', r'$g:a_w$'), ('g_bw', r'$g:b_w$')]))
    var_names[0].append(OrderedDict([('sig_x', r'$\sigma_x$'), ('sig_y', r'$\sigma_y$'),
                                     ('sig_xl', r'local $\sigma_x$'),
                                     ('sig_yl', r'local $\sigma_y$')]))
    var_names[0].append(OrderedDict([('corr', r'$\rho$'), ('corr_l', r'local $\rho$'),
                                     ('lam', r'$\lambda$')]))
    var_names.append([OrderedDict([('xi', r'$\xi$'), ('em_obs_prob', r'emerg obs prob'),
                                   ('grid_obs_prob', r'grid obs prob')])])
    sent_names = []
    for name in db.trace_names[0]:
        if name[:13] == 'sent_obs_prob':
            id = name[-1]
            sent_names.append((name, id))
    var_names[1].append(OrderedDict(sent_names))

    plt.figure()
    plt.hold(True)

    f_clrs = [0.3, 0.7]
    g_clrs = [0.1, 0.9]
    sig_clrs = [0.01, 0.99, 0.25, 0.75]
    corr_lam_clrs = [0.01, 0.25, 0.5]
    probs_clrs = [0.01, 0.5, 0.99]
    clrs_list = [f_clrs, f_clrs+g_clrs, sig_clrs, corr_lam_clrs, probs_clrs]

    plt.title("Traces of unknown model parameters")
    for ii in range(len(var_names[0])):
        plt.subplot(len(var_names[0]), 1, ii+1)
        cnt = 0
        for name, label in var_names[0][ii].items():
            plt.plot(db.trace(name, chain=None)[:], label="trace of "+label,
                     c=cmap(clrs_list[ii][cnt]), lw=lw)
            cnt += 1
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
    cnt = 0
    for name, label in var_names[1][0].items():
        plt.plot(db.trace(name, chain=None)[:], label="trace of "+label,
                 c=cmap(probs_clrs[cnt]), lw=lw)
        cnt += 1
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    # sent_obs_probs
    plt.subplot(212)
    cnt = 0
    for name, label in var_names[1][1].items():
        plt.plot(db.trace(name, chain=None)[:], label="trace of prob field "+label,
                 c=cmap(.10+cnt*.16), lw=lw)
        cnt += 1
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    plt.draw()

    ##### Convergence tests #####

    # Geweke
    f, axarr = plt.subplots(len(var_names[0]), sharex=True)
    axarr[0].set_title('Geweke Plots')
    axarr[0].hold(True)
    for ii in range(len(var_names[0])):
        cnt = 0
        ymax = 0
        ymin = 0
        for name, label in var_names[0][ii].items():
            scores = pm.geweke(db.trace(name, chain=None)[:])
            x, y = np.transpose(scores)
            axarr[ii].scatter(x.tolist(), y.tolist(), label=label,
                        c=cmap(clrs_list[ii][cnt]))
            ymax = max(ymax, np.max(y))
            ymin = min(ymin, np.min(y))
            cnt += 1
        # Legend
        leg = axarr[ii].legend(loc="upper left",prop={'size':9})
        leg.get_frame().set_alpha(0.7)
        # Labels
        axarr[ii].set_ylabel('Z-score')
        # Plot lines at +/- 2 std from zero
        axarr[ii].plot((np.min(x), np.max(x)), (2, 2), '--')
        axarr[ii].plot((np.min(x), np.max(x)), (-2, -2), '--')
        # Plot bounds
        axarr[ii].set_ylim(min(-2.5, ymin), max(2.5, ymax))
        axarr[ii].set_xlim(0, np.max(x))
    axarr[-1].set_xlabel('First iteration')

    plt.hold(False)
    if not os.path.exists(path):
        os.mkdir(path)
    if not path.endswith('/'):
        path += '/'
    plt.savefig("{}.{}".format(path+'_Geweke',format),dpi=200)
    plt.draw()



def plot_f_g(db=db, start=0, stop=None):
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
    plt.hist(db.trace("f_a1",chain=None)[start:stop], histtype='stepfilled', bins=27,
             alpha=0.85, label=r"posterior of position param $f_a1$",
             color=clrdict['color'], normed=True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("f_a2",chain=None)[start:stop], histtype='stepfilled', bins=27,
             alpha=0.85, label=r"posterior of position param $f_a2$",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper center")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(312) #f-b_1, f-b_2
    plt.xlim(1,15)
    plt.hold(True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("f_b1",chain=None)[start:stop], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $b_1$",
             color=clrdict['color'], normed=True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("f_b2",chain=None)[start:stop], histtype='stepfilled', bins=25,
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
    plt.hist(db.trace("g_aw",chain=None)[start:stop], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of position param $g_aw$",
             color=clrdict['color'], normed=True)
    unused = next(color_cycle)
    clrdict = next(color_cycle)
    plt.hist(db.trace("g_bw",chain=None)[start:stop], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior of shape param $g_bw$",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)



def plot_sprd_vars(db=db,start=0,stop=None):
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
    plt.hist(db.trace("sig_x",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior of wind $\sigma_x$",
             normed=True)
    plt.hist(db.trace("sig_y",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior of wind $\sigma_y$",
             normed=True)
    plt.hold(False)
    plt.xlim(0,300)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(412)
    plt.hold(True)
    plt.hist(db.trace("sig_xl",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior of local $\sigma_x$",
             normed=True)
    plt.hist(db.trace("sig_yl",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior of local $\sigma_y$",
             normed=True)
    plt.hold(False)
    plt.xlim(0,300)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(413)
    color_cycle = ax._get_lines.prop_cycler
    # get some new colors
    unused = next(color_cycle)
    unused = next(color_cycle)
    clrdict = next(color_cycle)
    plt.hold(True)
    plt.hist(db.trace("corr",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior of wind $\rho$",
             color=clrdict['color'], normed=True)
    clrdict = next(color_cycle)
    plt.hist(db.trace("corr_l",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior of local $\rho$",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    plt.xlim(-1,1)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(414)
    color_cycle = ax._get_lines.prop_cycler
    # get a new color
    for ii in range(3):
        unused = next(color_cycle)
    clrdict = next(color_cycle)
    # this is discrete data. need to bin it correctly
    tr = db.trace("n_periods",chain=None)[start:stop]
    plt.hold(True)
    plt.hist(tr, bins=np.arange(tr.min(),tr.max()+2,1)-.5,
             histtype='stepfilled',  alpha=0.85,
             label=r"posterior of avg flight time (min)",
             color=clrdict['color'], normed=True)
    plt.hold(False)
    plt.xlim(0,80)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)



def plot_sent_obs_probs(db=db,start=0,stop=None):
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
        plt.hist(db.trace(field_names[ii],chain=None)[start:stop],
                 histtype='stepfilled', bins=25, alpha=0.85,
                 label="field {}".format(field_ids[ii]),
                 normed=True)
        leg = plt.legend(loc="upper right")
        leg.get_frame().set_alpha(0.7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)



def plot_other(db=db,start=0,stop=None):
    '''Plot posteriors for lambda, xi, grid_obs_prob, em_obs_prob and A_collected

    Arguments:
        db: database object
        start: where to begin in the trace (with all chains taken together)
    '''

    plt.figure()
    ax = plt.subplot(411)
    plt.title(r"Posteriors for $\lambda$, $\xi$, grid_obs_prob and em_obs_prob")
    plt.hist(db.trace("lam",chain=None)[start:stop], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior for $\lambda$", normed=True)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(412)
    plt.hist(db.trace("xi",chain=None)[start:stop], histtype='stepfilled', bins=25,
             alpha=0.85, label=r"posterior for $\xi$", normed=True)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(413)
    plt.hold(True)
    plt.hist(db.trace("grid_obs_prob",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior for grid_obs_prob",
             normed=True)
    plt.hist(db.trace("em_obs_prob",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label=r"posterior for em_obs_prob",
             normed=True)
    plt.hold(False)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    ax = plt.subplot(414)
    plt.hist(db.trace("A_collected",chain=None)[start:stop], histtype='stepfilled',
             bins=25, alpha=0.85, label="posterior for A_collected", normed=True)
    leg = plt.legend(loc="upper right")
    leg.get_frame().set_alpha(0.7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)



if __name__ == "__main__":
    nargs = len(sys.argv) - 1
    if nargs == 0:
        print("Please include the database name as an argument, optionally\n"+
              "followed by the module to run (with or w/o start and stop point.")
    elif nargs > 0:
        database_name = sys.argv[1]
        if database_name[-3:] != '.h5':
            database_name += '.h5'
        if os.path.isfile(database_name):
            db = pm.database.hdf5.load(database_name)
        else:
            print("Invalid filename: {}".format(database_name))
            nargs = 0

    if nargs > 1:
        '''Run the requested plot'''
        if nargs == 3:
            start = int(sys.argv[3])
        elif nargs > 3:
            start = int(sys.argv[3])
            stop = int(sys.argv[4])
        else:
            start = 0
            stop = None

        if sys.argv[2] == 'plot_traces':
            plot_traces(db)
        elif sys.argv[2] == 'plot_f_g':
            plot_f_g(db,start,stop)
        elif sys.argv[2] == 'plot_sprd_vars':
            plot_sprd_vars(db,start,stop)
        elif sys.argv[2] == 'plot_sent_obs_probs':
            plot_sent_obs_probs(db,start,stop)
        elif sys.argv[2] == 'plot_other':
            plot_other(db,start,stop)
        else:
            print('Method not found.')
        input("Press Enter to finish...")
    elif nargs == 1:
        def get_args(strin):
            args = strin[1:].strip().split()
            if len(args) == 1:
                args.append(None)
            else:
                args[1] = int(args[1])
            args[0] = int(args[0])
            return args
        while True:
            '''Open an interactive menu'''
            print("----------Plot MCMC Results----------")
            print("(1) Plot traces")
            print("(2) Plot f & g argument posteriors")
            print("(3) Plot diffusion posteriors")
            print("(4) Plot sentinel field posteriors")
            print("(5) Plot others")
            print("(6) Quit")
            print("2-5 may be followed by a start number and a stop number,\n"+
                  "separted by a space.")
            cmd = input(":")

            try:
                if cmd[0] == "1":
                    plot_traces(db)
                elif cmd[0] == "2":
                    if cmd[1:].strip() == '':
                        plot_f_g(db)
                    else:
                        plot_f_g(db,*get_args(cmd))
                elif cmd[0] == "3":
                    if cmd[1:].strip() == '':
                        plot_sprd_vars(db)
                    else:
                        plot_sprd_vars(db,*get_args(cmd))
                elif cmd[0] == "4":
                    if cmd[1:].strip() == '':
                        plot_sent_obs_probs(db)
                    else:
                        plot_sent_obs_probs(db,*get_args(cmd))
                elif cmd[0] == "5":
                    if cmd[1:].strip() == '':
                        plot_other(db)
                    else:
                        plot_other(db,*get_args(cmd))
                elif cmd[0] == "6" or cmd[0] == "q" or cmd[0] == "Q":
                    break
                else:
                    print("Command not found.")
            except ValueError:
                print("Could not parse start number {}.".format(cmd[1:].strip()))