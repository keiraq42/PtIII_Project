import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pySALEPlot as psp

# Create output directory
dir = 'TrPMat'
psp.mkdir_p(dir)

# Open data files 
m = psp.opendatfile('../output/jdata.dat',scale='cm')

#Create subplots (1 row, 2 columns)
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(133, aspect = 'equal')
cbx = fig.add_axes([0.39, 0.11,0.03,0.77])
ccx = fig.add_axes([0.54,0.11,0.03,0.77])

# Loop over timesteps
for i in np.arange(0, m.nsteps, 10):

    # Set axis limits
    ax1.set_xlim(0,40)
    ax2.set_xlim([-m.xhires[1], m.xhires[1]])
    
    ax1.set_ylim([-3.5, 1])
    ax2.set_ylim([-3.5, 1])
    
    
    
    ax1.tick_params(axis='both', labelsize=7)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.set_xticks(np.arange(-3, 3.1, 1))
    ax2.set_yticks(np.arange(-3, 1.1, 1))
    ax1.title.set_text('Pressure')
    
    
    # Set axis labels
    ax1.set_xlabel('P [GPa]', fontsize=7)
    ax1.set_ylabel('y [cm]', fontsize=7)



    ax2.set_xlabel('x [cm]', fontsize=7)
    ax2.set_ylabel('y [cm]', fontsize=7)
    ax2.title.set_text(' Peak    Material')
    
    


    # Read Density and Pressure at this timestep
    s = m.readStep(['Pre', 'TrP'],i)
    fig.suptitle('Current vs Peak Pressure at {:3.2f} $\mu$s'.format(s.time*1e6),x=0.5)
    # Use figure title for time

    ax1.plot(s.Pre[50,:]*1e-9, m.yc[50,:], 'r--', lw=0.75, label = 'Transient Pressure Profile')
    
    PeakPressures = s.TrP*1.e-9
    ylocations = s.ymark
    
    indices = np.argsort(ylocations)
    
    PeakPressures = PeakPressures[indices]
    ylocations = ylocations[indices]
    
    top = np.amax(ylocations)
    bottom = np.amin(ylocations)
    intervals = np.linspace(bottom, top, num=100)
    
    depth = []
    avg_press = []
    max_press = []
    min_press = []

    for j in np.arange(0,len(intervals)-1,1):
        b = intervals[j]
        t = intervals[j+1]
        
        depth.append((b+t)/2)
        
        press = PeakPressures[(ylocations > b) & (ylocations < t)]

        if len(press) > 0:
            avg_press.append(np.mean(press))
        
            # Check for NaN and Inf values before calculating the maximum
            if np.any(np.isnan(press)) or np.any(np.isinf(press)):
                max_press.append(np.nan)
                min_press.append(np.nan)
            else:
                max_press.append(np.amax(press))
                min_press.append(np.amin(press))
        else:
            avg_press.append(np.nan)
            max_press.append(np.nan)
            min_press.append(np.nan)
      
    ax1.plot(avg_press, depth, lw=0.75, label = 'Average Peak Pressure')
#    ax1.plot(max_press, depth, lw=0.75, label = 'Maximum Peak Pressure')
 #   ax1.plot(min_press, depth, lw=0.75, label = 'Minimum Peak Pressure') 
    
    ax1.legend(fontsize=6)
    
    cmap = colors.ListedColormap(['m', 'y', 'k', 'g', 'r'])
    bounds = np.arange(1.5,7,1)
    norm = colors.BoundaryNorm(bounds,cmap.N)
    
    
    p = ax2.pcolormesh(m.x, m.y, s.mat, cmap=cmap, norm=norm)
    q = ax2.contour(m.xc, m.yc, s.cmc[1], 1 ,colors='k',linewidths=1)
    q = ax2.contour(-m.xc, m.yc, s.cmc[1], 1 ,colors='k',linewidths=1)
    
    r = ax2.scatter(-s.xmark, s.ymark, c = s.TrP*1.e-9, vmin=0, vmax=20, s = 0.01, cmap='plasma')
  
    # A colorbar
    cb1=fig.colorbar(p, cax=ccx, orientation='vertical')
    cb1.set_label(' Tit Iron MQC', fontsize = 7)
    cb1.set_ticks([])

    cb2=fig.colorbar(r, cax=cbx, orientation='vertical')
    cb2.set_label('Pressure [GPa]', fontsize = 7)

    # Save the figure
    print ('{}/{:03d}.png'.format(dir,i))
    fig.savefig('{}/{:03d}.png'.format(dir,i), dpi=300)


    # Clear the plots ready for the next timestep
    ax1.cla()
    ax2.cla()
