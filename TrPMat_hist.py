import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import math
import pySALEPlot as psp

# Open data files
m = psp.opendatfile('../output/jdata.dat', scale='cm', tracermassvol=True)
s = m.readStep(['Pre', 'TrP', 'TrM'], 150)
istep = m.readStep(['Pre', 'TrP', 'TrM'], 0)

TrP = s.TrP * 1.e-9

max = 350

print (np.unique(s.TrM))

for i in np.unique(s.TrM):

    mat = i

    
    if mat == 2:
       continue
       
    if mat == 6:
        TrP = s.TrP * 1.e-9
        TrM = s.TrM
        vol = m.tracerVolume    
        QTrP = TrP[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QTrM = TrM[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QtracerVolume = vol[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        TrP = QTrP[QTrM == mat]
        calc_vol = QtracerVolume[QTrM == mat]
        weight = 1/ calc_vol

        TrP_min = 0
        TrP_max = max
        bins_no = int((TrP_max - TrP_min) / 2) + 1    
        label = 'Enstatite'
        color = 'slategrey'
        kde = gaussian_kde(TrP, weights = weight)

        x = np.linspace(TrP_min, TrP_max, 1000)
        kde_values = kde(x)*4

        plt.plot(x, kde_values, label=label, color = color)            
    
    if mat == 5:
        TrP = s.TrP * 1.e-9
        TrM = s.TrM
        vol = m.tracerVolume    
        QTrP = TrP[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QTrM = TrM[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QtracerVolume = vol[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        TrP = QTrP[QTrM == mat]
        calc_vol = QtracerVolume[QTrM == mat]
        weight = 1/ calc_vol

        TrP_min = 0
        TrP_max = max
        bins_no = int((TrP_max - TrP_min) / 2) + 1    
        label = 'Grossular garnet'
        color = 'darkred'
        kde = gaussian_kde(TrP, weights = weight)

        x = np.linspace(TrP_min, TrP_max, 1000)
        kde_values = kde(x)*4

        plt.plot(x, kde_values, label=label, color = color)            
        
    if mat == 4:
        TrP = s.TrP * 1.e-9
        TrM = s.TrM
        vol = m.tracerVolume    
        QTrP = TrP[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QTrM = TrM[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QtracerVolume = vol[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        TrP = QTrP[QTrM == mat]
        calc_vol = QtracerVolume[QTrM == mat]
        weight = 1/ calc_vol

        TrP_min = 0
        TrP_max = max
        bins_no = int((TrP_max - TrP_min) / 2) + 1      
        label = 'Oligoclase'
        color = 'dodgerblue'
        kde = gaussian_kde(TrP, weights = weight)

        x = np.linspace(TrP_min, TrP_max, 1000)
        kde_values = kde(x)*4

        plt.plot(x, kde_values, label=label, color = color)            
        
    if mat == 3:
        TrP = s.TrP * 1.e-9
        TrM = s.TrM
        vol = m.tracerVolume    
        QTrP = TrP[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QTrM = TrM[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        QtracerVolume = vol[(istep.ymark < -0.2525) & (istep.ymark > -0.7525)]
        TrP = QTrP[QTrM == mat]
        calc_vol = QtracerVolume[QTrM == mat]
        weight = 1/ calc_vol

        TrP_min = 0
        TrP_max = max
        bins_no = int((TrP_max - TrP_min) / 2) + 1    
        label = 'Quartz'
        color = 'silver'
        kde = gaussian_kde(TrP, weights = weight)

        x = np.linspace(TrP_min, TrP_max, 1000)
        kde_values = kde(x)*4

        plt.plot(x, kde_values, label=label, color = color)            
    
        

        
        
plt.xlabel('P (GPa)')
plt.xlim(0,max)
plt.ylabel('Frequency Density')
plt.title('Pressure Histogram')
plt.legend()

plt.savefig('PHist.png')
