import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import math
import pySALEPlot as psp

# Open data files
m = psp.opendatfile('../output/jdata.dat', scale='cm', tracermassvol=True)
s = m.readStep(['Pre', 'TrP', 'TrM'], 100)

s.TrP = s.TrP * 1.e-9
max = 40
print (np.unique(s.TrM))

for i in np.unique(s.TrM):

    mat = i

    TrP = s.TrP[s.TrM == mat]
    calc_vol = m.tracerVolume[s.TrM == mat]
    weight = 1/ calc_vol

    TrP_min = 0
    TrP_max = max
    bins_no = int((TrP_max - TrP_min) / 2) + 1
    
    if mat == 1:
        continue

    if mat == 2:
        continue

    if mat == 3:
        continue

    if mat == 4:
        continue
       
    if mat == 7:
        label = 'Clay'
        color = 'peru'
    
    if mat == 6:
        label = 'Quartz'
        color = 'silver'
        
    if mat == 5:
        label = 'Magnetite'
        color = 'k'
       
        
    kde = gaussian_kde(TrP, weights = weight)

    x = np.linspace(TrP_min, TrP_max, 1000)
    kde_values = kde(x)

    plt.plot(x, kde_values, label=label, color = color)
        
        
plt.xlabel('P (GPa)')
plt.xlim(0,max)
plt.ylabel('Frequency Density')
plt.title('Pressure Histogram')
plt.legend()

plt.savefig('PHist.png')
