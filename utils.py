import sys
import os
sys.path.append('./software/modelprocessing-master') # make less local structure dependent!
import convert_rhybrid
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
mpl.rcParams.update(mpl.rcParamsDefault)


def vlsv_to_h5(inpath, outdir, outfile):
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
        convert_rhybrid.convert_dataset(inpath, outdir + '/' + outfile)
    except Exception:
        print(traceback.format_exc())
        raise RuntimeError('Error in conversion!')
        
def plot_3d(x, y, z, vx, vy, vz, t, label, units, outdir, outfile):
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

    plt.figure(figsize = [20,8])
    plt.suptitle(label + ', t=' + str(t))

    plt.subplot(131)
    ax = plt.gca()
    ax.streamplot(y, z, vy[idx_xmin,:,:], vz[idx_xmin,:,:], color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    xlims = ax.get_xlim()
    ylims = ax.get_xlim()
    im = ax.imshow(v[idx_xmin,:,:], extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                   aspect="equal",origin="lower",
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
    ax.streamplot(x, z, vx[:,idx_ymin,:], vz[:,idx_ymin,:], color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    xlims = ax.get_xlim()
    ylims = ax.get_xlim()
    im = ax.imshow(v[:,idx_ymin,:], extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
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
    ax.streamplot(x, y, vx[:,:,idx_zmin], vy[:,:,idx_zmin], color = 'white', linewidth = 0.5) # skip some becuase there are a lot of points
    xlims = ax.get_xlim()
    ylims = ax.get_xlim()
    im = ax.imshow(v[:,:,idx_zmin], extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
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
    xlims = ax.get_xlim()
    ylims = ax.get_xlim()
    im = ax.imshow(v[idx_xmin,:,:], extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
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
    xlims = ax.get_xlim()
    ylims = ax.get_xlim()
    im = ax.imshow(v[:,idx_ymin,:], extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
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
    xlims = ax.get_xlim()
    ylims = ax.get_xlim()
    im = ax.imshow(v[:,:,idx_zmin], extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
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
