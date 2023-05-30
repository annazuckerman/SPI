import sys
import os
sys.path.append('./software/modelprocessing-master') # make less local structure dependent!
import convert_rhybrid
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
mpl.rcParams.update(mpl.rcParamsDefault)


def vlsv_to_h5(inpath, outdir, outfile, radius):
    """
    Convert an input .vlsv file to .h5 using H. Egan's modelprocessing package.
    NOTE: must first update the desired fields to convert in
        modelprocessing-master/misc/name_conversion.txt, 
        and then update the corresponding names in
        modelprocessing-master/convert_rhybrid.py
    ----------
    Parameters:
        inpath (str):  path to input .vslv file
        outdir (str):  desired path to output .h5 file
        outfile (str): desired output .h5 filename (will overwrite!)
    ----------
    Returns:
        None: creates a new .h5 file in the specified path
    """

    if not all(isinstance(item, str) for item in [inpath, outdir, outfile]):
        raise TypeError('Input and output filepaths and filenames ' +
                        ' must be strings.')

    if not os.path.exists(inpath):
        raise FileNotFoundError('Could not find input file at: ' + inpath)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if os.path.exists(outdir + '/' + outfile):
        os.remove(outdir + '/' + outfile) # maybe later make it rename or ask


    try:
        convert_rhybrid.convert_dataset(inpath, outdir + '/' + outfile, radius)
    except Exception:
        print(traceback.format_exc())
        raise RuntimeError('Error in conversion!')

def plot_3d(x, y, z, vx, vy, vz, t, label, units, plot_type, outdir, outfile):
    """
    Plot a 3D vector field variable.
    NOTE: assumes variable has expected output shape from RHybrid model.
        (x, y, and z are length 62 where index 30 is the closest to zero)
    ----------
    Parameters:
        x (array):     1D numpy array of x values
        y (array):     1D numpy array of y values
        z (array):     1D numpy array of z values
        vx (array):    3D numpy array of vector x-components to be plotted
        vy (array):    3D numpy array of vector y-components to be plotted
        vz (array):    3D numpy array of vector z-components to be plotted
        t (int):       number representing the timestep corresponding to
                       the data to be plotted
        label (str):   label for plot (quantity name to be plotted)
        units (str):   label for units of quantity to be plotted
        outdir (str):  path to save output plot
        outfile (str): filename to save output plot (will overwrite!)
    ----------
    Returns:
        None: saves output plots
    """

    if not (isinstance(outdir, str) and isinstance(outfile, str)):
        raise TypeError('Output filepath and name must be a string.')

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if os.path.exists(outdir + '/' + outfile):
        os.remove(outdir + '/' + outfile) # maybe later make it rename or ask  

    idx_xmin = np.argmin(np.abs(x)) # idx in x closest to zero
    idx_ymin = np.argmin(np.abs(y)) # idx in y closest to zero
    idx_zmin = np.argmin(np.abs(z)) # idx in z closest to zero

    v = np.sqrt(vx**2 + vy**2 + vz**2)

    plt.figure(figsize = [20,8], dpi = 200)
    plt.suptitle(label + ', t=' + str(t))

    plt.subplot(131)
    ax = plt.gca()
    if plot_type == 'stream':
        ax.streamplot(y, z, np.transpose(vy[idx_xmin,:,:]), np.transpose(vz[idx_xmin,:,:]), color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    elif plot_type == 'quiver':
        colors = np.transpose(vx[idx_xmin,::5,::5])
        quiv = ax.quiver(y[::5], z[::5], np.transpose(vy[idx_xmin,::5,::5]), np.transpose(vz[idx_xmin,::5,::5]), colors, angles='xy')#, pivot = 'middle')
        plt.colorbar(quiv, label=label + ', out of plane ' +  units, orientation='horizontal')
    xlims = np.array([y[0], y[-1]]) #ax.get_xlim()
    ylims = np.array([z[0], z[-1]]) #ax.get_ylim()
    im = ax.imshow(np.transpose(v[idx_xmin,:,:]), extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                   #aspect="equal",
                   #origin="lower",
                   vmin=np.nanmin(v), vmax=np.nanmax(v))
    ax.set_aspect('equal')
    plt.title(r'YZ Plane (x$\approx$0)')
    plt.xlabel('y [km]')
    plt.ylabel('z [km]')
    plt.colorbar(im, ax=ax,
                 label = label + ' ' +  units,
                 orientation='horizontal')
                 #norm=mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v)))

    plt.subplot(132)
    ax = plt.gca()
    #ax.streamplot(x, z, vx[30,:,:], vz[30,:,:], color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    if plot_type == 'stream':
        ax.streamplot(x, z, np.transpose(vx[:,idx_ymin,:]), np.transpose(vz[:,idx_ymin,:]), color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    elif plot_type == 'quiver':
        colors = np.transpose(vy[::5,idx_ymin,::5])
        quiv = ax.quiver(x[::5], z[::5], np.transpose(vx[::5,idx_ymin,::5]), np.transpose(vz[::5,idx_ymin,::5]), colors, angles='xy')#, pivot = 'middle')
        plt.colorbar(quiv, label=label + ', out of plane ' +  units, orientation='horizontal')
    xlims = np.array([x[0], x[-1]]) # ax.get_xlim()
    ylims = np.array([z[0], z[-1]]) # ax.get_ylim()
    im = ax.imshow(np.transpose(v[:,idx_ymin,:]), extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                   aspect="equal",origin="lower",
                   vmin=np.nanmin(v), vmax=np.nanmax(v))
    ax.set_aspect('equal')
    #plt.quiver(ymesh[30,::2,::2],zmesh[30,::2,::2],vy[30,::2,::2],vz[30,::2,::2]) # skip some becuase there are a lot of points
    plt.title(r'XZ Plane (y$\approx$0)')
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar(im, ax=ax,
                 label = label + ' ' +  units,
                 orientation='horizontal')
                 #norm=mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v)))

    plt.subplot(133)
    ax = plt.gca()
   # ax.streamplot(x, y, vx[30,:,:], vy[30,:,:], color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    if plot_type == 'stream':
        ax.streamplot(x, y, np.transpose(vx[:,:,idx_zmin]), np.transpose(vy[:,:,idx_zmin]), color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    elif plot_type == 'quiver':
        colors = np.transpose(vz[::5,::5,idx_zmin])
        quiv = ax.quiver(x[::5], y[::5], np.transpose(vx[::5,::5,idx_zmin]), np.transpose(vy[::5,::5,idx_zmin]), colors, angles='xy')#, pivot = 'middle')
        plt.colorbar(quiv, label=label + ', out of plane ' +  units, orientation='horizontal')
    xlims = np.array([x[0], x[-1]]) #ax.get_xlim()
    ylims = np.array([y[0], y[-1]]) #ax.get_ylim()
    im = ax.imshow(np.transpose(v[:,:,idx_zmin]), extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                   aspect="equal",origin="lower",
                   vmin=np.nanmin(v), vmax=np.nanmax(v))
    ax.set_aspect('equal')
    #plt.quiver(ymesh[30,::2,::2],zmesh[30,::2,::2],vy[30,::2,::2],vz[30,::2,::2]) # skip some becuase there are a lot of points
    plt.title(r'XY Plane (z$\approx$0)')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.colorbar(im, ax=ax,
                 label = label + ' ' +  units,
                 orientation='horizontal')
                 #norm=mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v)))

    plt.savefig(outdir + '/' + outfile)
    plt.show()

def plot_1d(x, y, z, v, t, label, units, outdir, outfile):
    """
    Plot a 1D variable.
    NOTE: assumes variable has expected output shape from RHybrid model.
        (x, y, and z are length 62 where index 30 is the closest to zero)
    ----------
    Parameters:
        x (array):     1D numpy array of x values
        y (array):     1D numpy array of y values
        z (array):     1D numpy array of z values
        v (array):     3D numpy array of quantity to be plotted
        t (int):       number representing the timestep corresponding to
                       the data to be plotted
        label (str):   label for plot (quantity name to be plotted)
        units (str):   label for units of quantity to be plotted
        outdir (str):  path to save output plot
        outfile (str): filename to save output plot (will overwrite!)
    ----------
    Returns:
        None: saves output plots
    """

    if not (isinstance(outdir, str) and isinstance(outfile, str)):
        raise TypeError('Output filepath and name must be a string.')

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if os.path.exists(outdir + '/' + outfile):
        os.remove(outdir + '/' + outfile) # maybe later make it rename or ask

    idx_xmin = np.argmin(np.abs(x)) # idx in x closest to zero
    idx_ymin = np.argmin(np.abs(y)) # idx in y closest to zero
    idx_zmin = np.argmin(np.abs(z)) # idx in z closest to zero

    plt.figure(figsize = [20,8])
    plt.suptitle(label + ', t=' + str(t))

    plt.subplot(131)
    ax = plt.gca()
    #xlims = ax.get_xlim()
    #ylims = ax.get_ylim()
    im = ax.imshow(np.transpose(v[idx_xmin,:,:]),# extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                   aspect="equal",origin="lower",
                   norm=colors.SymLogNorm(base = 10, vmin=np.nanmin(v), vmax=np.nanmax(v), linthresh = 0.0001*np.nanmax(v), linscale = 0.5))
    current_cmap = mpl.cm.get_cmap()
    current_cmap.set_bad(color='red')
    ax.set_aspect('equal')
    plt.title(r'YZ Plane (x$\approx$0)')
    plt.xlabel('y [km]')
    plt.ylabel('z [km]')
    plt.colorbar(im, ax=ax,
                 label = label + ' ' +  units,
                 orientation='horizontal')
                 #norm=mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v)))

    plt.subplot(132)
    ax = plt.gca()
    #xlims = ax.get_xlim()
    #ylims = ax.get_ylim()
    im = ax.imshow(np.transpose(v[:,idx_ymin,:]),# extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                   aspect="equal",origin="lower",
                   norm=colors.SymLogNorm(base = 10, vmin=np.nanmin(v), vmax=np.nanmax(v), linthresh = 0.0001*np.nanmax(v), linscale = 0.5))
    ax.set_aspect('equal')
    plt.title(r'XZ Plane (y$\approx$0)')
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar(im, ax=ax,
                 label = label + ' ' +  units,
                 orientation='horizontal')
                 #norm=mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v)))

    plt.subplot(133)
    ax = plt.gca()
    #xlims = ax.get_xlim()
    #ylims = ax.get_ylim()
    im = ax.imshow(np.transpose(v[:,:,idx_zmin]), #extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                   aspect="equal",origin="lower",
                   norm=colors.SymLogNorm(base = 10, vmin=np.nanmin(v), vmax=np.nanmax(v), linthresh = 0.0001*np.nanmax(v), linscale = 0.5))
    ax.set_aspect('equal')
    plt.title(r'XY Plane (z$\approx$0)')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.colorbar(im, ax=ax,
                 label = label + ' ' +  units,
                 orientation='horizontal')
                 #norm=mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v)))

    plt.savefig(outdir + '/' + outfile)
    plt.show()


def calc_flux(x, y, z, n, vx, vy, vz): # d,:
    """
    Calculate the flux of a quantity through a closed (cube) surface.
    NOTE: assumes variable has expected output shape from RHybrid model.
        (x, y, and z are length 62 where index 30 is the closest to zero)
    ----------
    Parameters:
        x (array):     1D numpy array of x values
        y (array):     1D numpy array of y values
        z (array):     1D numpy array of z values
        n (array):     3D numpy array of number density values
        vx (array):    3D numpy array of vector x-components to be plotted
        vy (array):    3D numpy array of vector y-components to be plotted
        vz (array):    3D numpy array of vector z-components to be plotted
 ##       d (int):       (perpendicular) distance to surface edges in number of cells
 ##       m (float):     mass of the particle [kg]
    ----------
    Returns:
        tot_outward_flux : total escaping flux at the timestep 
        flux_px:           escaping flux through plus x wall
        flux_py:           escaping flux through plus y wall
        flux_pz:           escaping flux through plus z wall
        flux_nx:           escaping flux through negative x wall
        flux_ny:           escaping flux through negative y wall
        flux_nz:           escaping flux through negative z wall
    """
    
    # CURRENTLY ONLY WORKS IF d = n_total_cells/2, because not cutting on the dimensions of the faces!!

    idx_xmin = np.argmin(np.abs(x)) # idx in x closest to center of planet
    idx_ymin = np.argmin(np.abs(y)) # idx in y closest to center of planet
    idx_zmin = np.argmin(np.abs(z)) # idx in z closest to center of planet
    
    # volume element (assumes constant)
    dx = np.abs((x[1]-x[0]))
    dy = np.abs((y[1]-y[0]))
    dz = np.abs((z[1]-z[0]))

    # wall normal to +x 
    #norm = np.array([1,0,0]) # unit normal to the yz plane
    #n_wall_px = np.ravel(n[idx_xmin + d,:,:]) # select the cells along the +x wall, flatten bc order doens't matter 
    #vx_wall_px = np.ravel(vx[idx_xmin + d,:,:])
    #vy_wall_x = np.ravel(vy[x_idx,:,:])
    #vz_wall_x = np.ravel(vz[x_idx,:,:])
    
    # wall normal to -x
    #n_wall_nx = np.ravel(n[idx_xmin - d,:,:]) # select the cells along the +x wall, flatten bc order doens't matter 
    #vx_wall_nx = np.ravel(vx[idx_xmin - d,:,:])

    # wall normal to +y
    #n_wall_py = np.ravel(n[idx_ymin + d,:,:]) 
    #vx_wall_py = np.ravel(vx[idx_ymin + d,:,:])
    
    # wall normal to -y
    #n_wall_ny = np.ravel(n[idx_ymin - d,:,:]) 
    #vx_wall_ny = np.ravel(vx[idx_ymin - d,:,:])

    # wall normal to +z
    #n_wall_pz = np.ravel(n[idx_zmin + d,:,:]) 
    #vx_wall_pz = np.ravel(vx[idx_zmin + d,:,:])
    
    # wall normal to -z
    #n_wall_nz = np.ravel(n[idx_zmin - d,:,:]) 
    #vx_wall_nz = np.ravel(vx[idx_zmin - d,:,:])
    
    #  FOR NOW USING THE SIMULATION BOUNDS AS THE BOX
    # wall normal to +x 
    n_wall_px = np.ravel(n[-1,:,:]) # select the cells along the +x wall, flatten bc order doesn't matter 
    vx_wall_px = np.ravel(vx[-1,:,:])
    
    # wall normal to -x
    n_wall_nx = np.ravel(n[0,:,:]) # select the cells along the -x wall, flatten bc order doesn't matter 
    vx_wall_nx = np.ravel(vx[0,:,:])

    # wall normal to +y
    n_wall_py = np.ravel(n[:,-1,:]) 
    vy_wall_py = np.ravel(vy[:,-1,:])
    
    # wall normal to -y
    n_wall_ny = np.ravel(n[:,0,:]) 
    vy_wall_ny = np.ravel(vy[:,0,:])

    # wall normal to +z
    n_wall_pz = np.ravel(n[:,:,-1]) 
    vz_wall_pz = np.ravel(vz[:,:,-1])
    
    # wall normal to -z
    n_wall_nz = np.ravel(n[:,:,0]) 
    vz_wall_nz = np.ravel(vz[:,:,0])
    
    
    outward_flux_px = 0
    N_cells = len(n_wall_px) # currently assume it's a cube (all faces same size)
    
    # +x wall
    outward_flux_px = 0
    for i in range(N_cells):
        dflux = n_wall_px[i]*vx_wall_px[i]*dz*dy # outwards flux per cell [#/s]
        outward_flux_px += dflux
            
    # +y wall
    outward_flux_py = 0
    for i in range(N_cells):
        dflux = n_wall_py[i]*vy_wall_py[i]*dx*dz # outwards flux per cell [#/s]
        outward_flux_py += dflux

    # +z wall
    outward_flux_pz = 0
    for i in range(N_cells):
        dflux = n_wall_pz[i]*vz_wall_pz[i]*dx*dy # outwards flux per cell [#/s]
        outward_flux_pz += dflux

    # -x wall
    outward_flux_nx = 0
    for i in range(N_cells):
        dflux = -1*n_wall_nx[i]*vx_wall_nx[i]*dz*dy # outwards flux per cell [#/s]
        outward_flux_nx += dflux
        
    # -y wall
    outward_flux_ny = 0
    for i in range(N_cells):
        dflux = -1*n_wall_ny[i]*vy_wall_ny[i]*dx*dz # outwards flux per cell [#/s]
        outward_flux_ny += dflux

    # -z wall
    outward_flux_nz = 0
    for i in range(N_cells):
        dflux = -1*n_wall_nz[i]*vz_wall_nz[i]*dx*dy # outwards flux per cell [#/s]
        outward_flux_nz += dflux
     
    tot_outward_flux = outward_flux_px + outward_flux_py + outward_flux_pz + outward_flux_nx + outward_flux_ny + outward_flux_nz

    return tot_outward_flux, outward_flux_px, outward_flux_py, outward_flux_pz, outward_flux_nx, outward_flux_ny, outward_flux_nz
