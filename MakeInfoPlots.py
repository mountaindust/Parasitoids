#! /usr/bin/env python3
'''This module is for generating publication quality plots that give general
info about the experiental setup and model diffusion behaviors.

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Run import Params
import ParasitoidModel as PM
from Plot_Result import get_satellite
from Data_Import import LocInfo

def main(bw=False):
    '''Spit out two plots, one of the sentinel field locations, labeled, with
    the average wind direction, and the other with a visual of wind and local
    diffusion as based on parameters listed in Run.py'''

    params = Params()
    locinfo = LocInfo(params.dataset,params.coord,params.domain_info)

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    ### Field locations ###
    domain_info = (3500,140) # make the domain 12x12 km, 25x25m resolution
    cell_dist = domain_info[0]/domain_info[1]
    xmesh = np.linspace(-domain_info[0]-cell_dist/2,
        domain_info[0]+cell_dist/2,domain_info[1]*2+2)
    plot_limits = [xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]]
    ax.axis(plot_limits)
    # get satellite image
    sat_img = get_satellite(params.maps_key,params.maps_service,
            params.coord,xmesh[-1])
    if sat_img is not None:
        if not bw: #color
            ax.imshow(sat_img,zorder=0,extent=plot_limits)
        else: #black and white
            sat_img = sat_img.convert('L') #B/W satellite image w/ dither
            ax.imshow(sat_img,zorder=0,cmap=plt.get_cmap('gray'),
                        extent=plot_limits)
    # field locations. Also add labels based at a position based on info in
    #   each Path object
    if not bw: #color
        for id,poly in locinfo.field_polys.items():
            ax.add_patch(patches.PathPatch(poly,facecolor='none',
                            edgecolor='r',lw=2,zorder=1))
            ext = poly.get_extents()
            # Put the label somewhere in the middle of each field
            ax.text((ext.xmin+ext.xmax)/2,(ext.ymin+ext.ymax)/2,id,
                    fontsize=12,color='w')
    else: #black and white
        for id,poly in locinfo.field_polys.items():
            ax.add_patch(patches.PathPatch(poly,facecolor='none',
                            edgecolor='k',lw=2,zorder=1))
            ext = poly.get_extents()
            # Put the label somewhere in the middle of each field
            ax.text((ext.xmin+ext.xmax)/2,(ext.ymin+ext.ymax)/2,id,
                    fontsize=12,color='w')
    # get average wind direction
    day_avg = []
    wind_data,days = PM.get_wind_data(*params.get_wind_params())
    for day in days:
        day_avg.append(wind_data[day].mean(axis=0))
    day_avg = np.array(day_avg)
    avg_wind = day_avg.mean(axis=0)
    # draw arrow
    ax.arrow(0,0,1600*avg_wind[0],1600*avg_wind[1],
             head_width=250,head_length=375,fc='0.8',ec='0.8')
    # get theta for avg wind direction
    if (avg_wind[0] == 0) and (avg_wind[1] == 0):
        theta = 0
    elif (avg_wind[0] == 0) and (avg_wind[1] > 0):
        theta = np.pi/2
    elif (avg_wind[0] == 0) and (avg_wind[1] < 0):
        theta = -np.pi/2
    else:
        theta = np.arctan(avg_wind[1]/avg_wind[0])
    if avg_wind[0] < 0:
        theta = theta+np.pi
    theta = int(np.degrees(theta))
    # show info
    ax.text(1600*avg_wind[0]-200,1600*avg_wind[1]+500,
        'Avg. wind speed/direction:\n'+
        '{:.1f} km/hr\n{} degrees from east'.format(avg_wind[2],theta),
        color='w',fontsize=11)
    ax.set_xlabel('West-East (meters)')
    ax.set_ylabel('North-South (meters)')
    # label plot
    ax.text(0.01,0.95,'a)',color='w',ha='left',va='center',
            transform=ax.transAxes,fontsize=16)

    ### Diffusion visualization ###
    ax = fig.add_subplot(122)
    # Generate diffusion clouds based on parameters in params
    # wind-based diffusion
    cov_wind = PM.Dmat(*params.Dparams)
    cloud_wind = np.random.multivariate_normal((0,0),cov_wind,1000)
    cov_local = PM.Dmat(*params.Dlparams)
    cloud_local = np.random.multivariate_normal((0,0),cov_local,100)
    # plot
    dist = max(params.Dparams[0],params.Dparams[1])*4
    plot_limits = [-dist,dist,-dist,dist]
    ax.axis(plot_limits)
    ax.scatter(cloud_wind[:,0],cloud_wind[:,1],s=5,c='0.85',zorder=1,
        edgecolors='none',linewidths=1,label='Wind-based diffusion')
    ax.scatter(cloud_local[:,0],cloud_local[:,1],s=2,c='k',zorder=2,
        edgecolors='none',label='Local diffusion')
    leg = ax.legend(loc="upper right",fontsize=11,framealpha=0.2,fancybox=True,
                    markerscale=2)
    for txt in leg.get_texts():
        txt.set_color('w')
    # add satellite image
    sat_img = get_satellite(params.maps_key,params.maps_service,
            params.coord,dist)
    if sat_img is not None:
        if not bw: #color
            ax.imshow(sat_img,zorder=0,extent=plot_limits)
        else: #black and white
            sat_img = sat_img.convert('L') #B/W satellite image w/ dither
            ax.imshow(sat_img,zorder=0,cmap=plt.get_cmap('gray'),
                        extent=plot_limits)
    ax.set_xlabel('West-East (meters)')
    ax.set_ylabel('North-South (meters)')
    # label plot
    ax.text(0.01,0.95,'b)',color='w',ha='left',va='center',
            transform=ax.transAxes,fontsize=16)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()