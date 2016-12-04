'''This module contains supporting functions for Bayes_model.
It includes functions to convert population density to expected emergence.

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

import numpy as np

### Oviposition to emergence time is definied here ###
# Specify the maximum incubation time and a numpy array. The last entry in the
#   numpy array is assumed to correspond to the max incubation time, so that
#   max_incubation_time - (incubation_time.size - 1) would give you the minimum
#   incubation time.
# We will assume incubation is 19 to 25 days, distributed approximately
#   according to a normal distribution with variance of 2.
incubation_time = np.array([0.05,0.1,0.2,0.3,0.2,0.1,0.05]) #19-25 inclusive
max_incubation_time = 25

def popdensity_to_emergence(modelsol,locinfo):
    '''Translate population model to corresponding expected number of wasps in
    a given location whose oviposition would result in a given emergence date.
    Only use the locations in which data was actually collected.
    '''

    # Assume collections are done at the beginning of the day, observations
    #   of collection data at the end of the day. So, oviposition is not possible
    #   on the day of collection, but emergence is.

    ### Project release field grid ###
    release_emerg = []
    for nframe,dframe in enumerate(locinfo.release_DataFrames):
        # Each dataframe should be sorted already, 'datePR','row','column'.
        # Also, the grid for each collection is stored in the list
        #   locinfo.emerg_grids.

        collection_day = (locinfo.collection_datesPR[nframe]).days

        ### Find the earliest and latest oviposition date PR that we need to ###
        ### simulate for this collection. 0 = release day.                   ###
        # The last day oviposition is possible is the day before collection
        # The earliest day oviposition is possible is the max incubation time
        #   before the first possible emergence
        start_day = max(collection_day - max_incubation_time,0) # days post release!
        ########################################################################

        #
        # Go through each feasible oviposition day of the model, projecting emergence
        #

        # emerg_proj holds each grid point in its rows and a different emergence
        #   day in its columns.
        # Feasible emergence days span the maximum incubation time.
        emerg_proj = np.zeros((len(locinfo.emerg_grids[nframe]),
            max_incubation_time))

        # go through feasible oviposition days
        for day in range(start_day,collection_day):
            n = 0 # row/col count
            # in each one, go through grid points projecting emergence date
            #   potentials per adult wasp per cell.
            max_post_col = day+max_incubation_time-collection_day
            min_post_col = max(0,max_post_col+1-incubation_time.size)
            span_len = max_post_col-min_post_col+1
            for r,c in locinfo.emerg_grids[nframe]:
                ###                Project forward and store                 ###
                ### This function is a mapping from feasible                 ###
                ###   oviposition dates to array of feasible emergence dates ###
                # day represents feasible oviposition days, [start,collect)
                e_distrib = modelsol[day][r,c]*incubation_time
                emerg_proj[n,min_post_col:max_post_col+1] += e_distrib[-span_len:]
                # time is now measured in days post collection
                ################################################################
                n += 1

        # now consolidate these days into just the days data was collected.
        # first, get unique dates
        obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
        modelsol_grid_emerg = np.zeros((len(locinfo.emerg_grids[nframe]),
                                        len(obs_datesPR)))
        col_indices = obs_datesPR - collection_day # days post collection
        # each observation day includes emergences on the day itself and any days
        #   since the last observation day.
        modelsol_grid_emerg[:,0] = emerg_proj[:,0:col_indices[0]+1].sum(axis=1)
        for n,col in enumerate(col_indices[1:]):
            col_last = col_indices[n]
            modelsol_grid_emerg[:,n+1] = emerg_proj[:,col_last+1:col+1].sum(axis=1)
        # we will lose any projected emergences past the last observation date.
        release_emerg.append(modelsol_grid_emerg)

    ### Project sentinel field emergence ###
    sentinel_emerg = []
    for nframe,dframe in enumerate(locinfo.sent_DataFrames):
        # Each dataframe should be sorted already, 'datePR','id'

        collection_day = (locinfo.collection_datesPR[nframe]).days

        ### Find the earliest and latest oviposition date PR that we need to ###
        ### simulate for this collection. 0 = release day.                   ###
        # The last day oviposition is possible is the day before collection
        # The earliest day oviposition is possible is the max incubation time
        #   before the first possible emergence
        start_day = max(collection_day - max_incubation_time,0) # days post release!
        ########################################################################

        #
        # Go through each feasible oviposition day of the model, projecting emergence
        #

        # emerg_proj holds each sentinel field in its rows and a different
        #   emergence day in its columns.
        # Feasible emergence days start at collection and go until observation stopped
        emerg_proj = np.zeros((len(locinfo.sent_ids),
            max_incubation_time))

        # go through feasible oviposition days
        for day in range(start_day,collection_day):
            # for each day, aggregate the population in each sentinel field
            max_post_col = day+max_incubation_time-collection_day
            min_post_col = max(0,max_post_col+1-incubation_time.size)
            span_len = max_post_col-min_post_col+1 #correct fencepost error
            for n,field_id in enumerate(locinfo.sent_ids):
                ###     Sum the field cells, project forward and store       ###
                ### This function can be more complicated if we want to try  ###
                ###   and be more precise. It's a mapping from feasible      ###
                ###   oviposition dates to array of feasible emergence dates ###
                field_total = modelsol[day][locinfo.field_cells[field_id][:,0],
                                    locinfo.field_cells[field_id][:,1]].sum()
                e_distrib = field_total*incubation_time
                emerg_proj[n,min_post_col:max_post_col+1] += e_distrib[-span_len:]
                ################################################################

        # now consolidate these days into just the days data was collected.
        # first, get unique dates
        obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
        modelsol_field_emerg = np.zeros((len(locinfo.sent_ids),
                                        len(obs_datesPR)))
        col_indices = obs_datesPR - collection_day # days post collection
        modelsol_field_emerg[:,0] = emerg_proj[:,0:col_indices[0]+1].sum(axis=1)
        for n,col in enumerate(col_indices[1:]):
            col_last = col_indices[n]
            modelsol_field_emerg[:,n+1] = emerg_proj[:,col_last+1:col+1].sum(axis=1)
        sentinel_emerg.append(modelsol_field_emerg)

    ### This process results in two lists, release_emerg and sentinel_emerg.
    ###     Each list entry corresponds to a data collection day (one array)
    ##      In each array:
    ###     Each column corresponds to an emergence observation day (as in data)
    ###     Each row corresponds to a grid point or sentinel field, respectively
    ### This format will need to match a structured data arrays for comparison

    return (release_emerg,sentinel_emerg)



def popdensity_grid(modelsol,locinfo):
    '''Translate population model to corresponding expected number of wasps in
    each grid point
    FUTURE: Make this wasps per m**2 instead, so this variable scales with
    different cell sizes.
    '''

    # Assume observations are done at the beginning of the day.
    grid_counts = np.zeros((locinfo.grid_cells.shape[0],
                            len(locinfo.grid_obs_datesPR)))

    for nday,date in enumerate(locinfo.grid_obs_datesPR):
        n = 0 # row/col count
        # for each day, get expected population at each grid point
        for r,c in locinfo.grid_cells:
            # model holds end-of-day PR results
            grid_counts[n,nday] = modelsol[date.days-1][r,c]
            n += 1

    ### Return ndarray where:
    ###     Each column corresponds to an observation day
    ###     Each row corresponds to a grid point

    return grid_counts



def popdensity_card(modelsol,locinfo,domain_info):
    '''Translate population model to expected number of wasps in cardinal
    directions.
    '''

    # Assume observations are done at the beginning of the day.
    # Form a list, one entry per sample day
    card_counts = []
    res = domain_info[0]/domain_info[1] # cell length in meters

    for nday,date in enumerate(locinfo.card_obs_datesPR):
        # get array shape
        obslen = locinfo.card_obs[nday].shape[1]
        day_count = np.zeros((4,obslen))
        # the release point is an undisturbed 5x5m area from which all
        #   sampling distances are calculated.
        dist = 5
        for step in range(obslen):
            dist += locinfo.step_size[nday]
            cell_delta = int(dist//res)
            # north
            day_count[0,step] = modelsol[date.days-1][domain_info[1]-cell_delta,
                                                      domain_info[1]]
            # south
            day_count[1,step] = modelsol[date.days-1][domain_info[1]+cell_delta,
                                                      domain_info[1]]
            # east
            day_count[2,step] = modelsol[date.days-1][domain_info[1],
                                                      domain_info[1]+cell_delta]
            # west
            day_count[3,step] = modelsol[date.days-1][domain_info[1],
                                                      domain_info[1]-cell_delta]
        card_counts.append(day_count)

    ### Return list of ndarrays where:
    ###     Each column corresponds to step in the cardinal direction
    ###     Each row corresponds to a cardinal direction: north,south,east,west

    return card_counts
