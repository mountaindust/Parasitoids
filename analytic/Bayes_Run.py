#! /usr/bin/env python3

'''This module uses PyMC to fit parameters to the model via Bayesian inference.

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.3"
__copyright__ = "Copyright 2015, Christopher Strickland"

import sys, time
import os.path
import numpy as np
import pymc as pm
import Bayes_model
import IPython

###############################################################################
#                                                                             #
#                             PYMC Setup & Run                                #
#                                                                             #
###############################################################################

def main():

    #####       Start Interactive Menu      #####
    print('--------------- MCMC MAIN MENU ---------------')
    print(" 'new': Start a new MCMC chain from the beginning.")
    print("'cont': Continue a previous MCMC chain from an hdf5 file.")
    #print("'plot': Plot traces/distribution from an hdf5 file.")
    print("'quit': Quit.")
    cmd = input('Enter: ')
    cmd = cmd.strip().lower()
    if cmd == 'new':
        print('\n\n')
        print('--------------- New MCMC Chain ---------------')
        while True:
            val = input("Enter number of realizations or 'quit' to quit:")
            val = val.strip()
            if val == 'q' or val == 'quit':
                return
            else:
                try:
                    nsamples = int(val)
                    val2 = input("Enter number of realizations to discard:")
                    val2 = val2.strip()
                    if val2 == 'q' or val2 == 'quit':
                        return
                    else:
                        burn = int(val2)
                    fname = input("Enter filename to save or 'back' to cancel:")
                    fname = fname.strip()
                    if fname == 'q' or fname == 'quit':
                        return
                    elif fname == 'b' or fname == 'back':
                        continue
                    else:
                        fname = fname+'.h5'
                        break # BREAK LOOP AND RUN MCMC WITH GIVEN VALUES
                except ValueError:
                    print('Unrecognized input.')
                    continue
        ##### RUN FIRST MCMC HERE #####
        mcmc = pm.MCMC(Bayes_model,db='hdf5',dbname=fname,
                        dbmode='a',dbcomplevel=0)
        try:
            mcmc.sample(nsamples,burn)
            # sampling finished. commit to database and continue
            mcmc.save_state()
            mcmc.commit()
        except:
            print('Exception: database closing...')
            mcmc.db.close()
            raise

    elif cmd == 'cont':
        # Load db and continue
        print('\n')
        while True:
            fname = input("Enter path to database to load, or 'q' to quit:")
            fname = fname.strip()
            if fname.lower() == 'q' or fname.lower() == 'quit':
                return
            else:
                if fname[-3:] != '.h5':
                    fname += '.h5'
                if os.path.isfile(fname):
                    db = pm.database.hdf5.load(fname)
                    mcmc = pm.MCMC(Bayes_model,db=db)
                    break # database loaded
                else:
                    print('File not found.')
                    #continue

    elif cmd == 'plot':
        # Get filename and pass to plotting routine.
        pass
        # return
    elif cmd == 'quit' or cmd == 'q':
        return
    else:
        print('Command not recognized.')
        print('Quitting....')
        return
        
    ##### MCMC Loop #####
    # This should be reached only by cmd == 'new' or 'cont' with a database.
    # It resumes sampling of a previously sampled chain.
    print('\n')
    while True:
        print('--------------- MCMC ---------------')
        print("'inspect': launch IPython to inspect state")
        print("    'run': conduct further sampling")
        print("   'quit': Quit")
        cmd = input('Enter: ')
        cmd = cmd.strip().lower()
        if cmd == 'inspect':
            try:
                IPython.embed()
            except:
                print('Exception: database closing...')
                mcmc.db.close()
                raise
        elif cmd == 'run':
            val = input("Enter number of realizations or 'back':")
            val = val.strip()
            if val == 'back' or val == 'b':
                continue
            else:
                try:
                    nsamples = int(val)
                except ValueError:
                    print('Unrecognized input.')
                    continue
            # Run chain
            try:
                mcmc.sample(nsamples)
                # sampling finished. commit to database and continue
                mcmc.save_state()
                mcmc.commit()
            except:
                print('Exception: database closing...')
                mcmc.db.close()
                raise
        elif cmd == 'quit' or cmd == 'q':
            mcmc.db.close()
            print('Database closed.')
            break
        else:
            print('Command not recognized.')
    
if __name__ == "__main__":
    main(sys.argv[1:])