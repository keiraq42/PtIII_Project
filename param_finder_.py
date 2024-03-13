import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from scipy import stats
import statistics
import os


# Get the desired filename from the user
filename = input("Enter the desired mineral:")

 #Search for the file with the provided name in the current directory and its subdirectories
found_file = None

for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.startswith(filename) and file.endswith('.xlsx'):
            found_file = os.path.join(root, file)
            break

if found_file:
    df = pd.read_excel(found_file)
    print (filename, "data found")
else:
    print(f"No Excel file with the name '{filename}' was found.")

columns1 = ["U", "D"]
df_v = pd.read_excel(found_file, usecols=columns1)
df_v = df_v.dropna()


x = np.array(df_v.U * 10**3)
y = np.array(df_v.D * 10 **3)
data = np.column_stack((x, y))

if filename == "Augite":
    filtered_data = data[data[:, 0] >= 500]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))

if filename == "Corundum":
    filtered_data = data[data[:, 0] <= 5000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))

if filename == "Diopside":
    filtered_data = data[data[:, 0] >= 500]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y)) 
    
if filename == "Hematite":
    filtered_data = data[data[:, 0] <= 1500]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y)) 
    
#if filename == "Kyanite":
 #   filtered_data = data[data[:, 0] >= 2000]
  #  filtered_x = filtered_data[:, 0]
   # filtered_y = filtered_data[:, 1]
    #data = np.column_stack((filtered_x, filtered_y))
    
if filename == "Magnetite":
    filtered_data = data[data[:, 0] <= 1000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))

if filename == "Oligoclase":
    filtered_data = data[data[:, 0] >= 210]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))    
    
if filename == "Orthoclase":
    filtered_data = data[data[:, 1] <= 6500]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))    
    
if filename == "Quartz, amorphous":
    #data = data[data[:, 0] >= 2000]
    filtered_x = data[:, 0]
    filtered_y = data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))
    
if filename == "Quartz, fused":
    filtered_data = data[data[:, 0] >= 2000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))    
    
if filename == "Quartz":
    filtered_data = data[data[:, 0] >= 2000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))

if filename == "Periclase":
    filtered_data = data[data[:, 0] <= 4000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))    
    
if filename == "Rutile":
    filtered_data = data[data[:, 0] >= 1000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))
    
if filename == "Serpentine":
    filtered_data = data[data[:, 0] <= 3000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))

if filename == "Sillimanite":
    filtered_data = data[data[:, 0] >= 2200]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))

if filename == "Stishovite":
    filtered_data = data[data[:, 1] >= 12000]
    filtered_x = filtered_data[:, 0]
    filtered_y = filtered_data[:, 1]
    data = np.column_stack((filtered_x, filtered_y))
    
#if filename == "Polycar":
 #   filtered_data = data[data[:, 0] <= 5000]
  #  filtered_x = filtered_data[:, 0]
   # filtered_y = filtered_data[:, 1]
    #data = np.column_stack((filtered_x, filtered_y))        
    
# Separate the x and y values from the filtered data
filtered_x = data[:, 0]
filtered_y = data[:, 1]

up = filtered_x
Udata = filtered_y

def U(up, C, S):
    U_predicted = C + S*up
    return U_predicted


bounds = ([0*10**3, 0], [8000, np.inf])
popt, pcov = scipy.optimize.curve_fit(U, up, Udata, bounds = bounds)
C = popt[0]
S = popt[1]
[C_error, S_error] = np.sqrt(np.diag(pcov))

#C = C *10**-3
C_error = C_error*10**-3

print ('C =', round(C*10**-3, 2), '+/-', round(C_error, 2))
print ('S =', round(popt[1], 2), '+/-', round(S_error, 2))


columnsR = ["R", "ratio"]
df_v = pd.read_excel(found_file, usecols=columnsR)
df_v = df_v.dropna()


R = np.array(df_v.R)
ratio = np.array(df_v.ratio)
rho0 = np.mean(R/ratio)

def A(C, rho0):
    A = rho0 * C**2
    return A

A_v = A(popt[0], rho0)
A_error = 2*(C_error)

predicted_values = U(up, (A_v/rho0)**0.5, S)
U_km = Udata * 10**-3

residuals = abs(U_km - predicted_values)
squared_residuals = residuals**2
mean_squared_error = np.mean(squared_residuals)
RMSE = np.sqrt(mean_squared_error)/np.mean(U_km)

print('RMS (A) =', round(RMSE, 2))

X = np.linspace(0, max(up), num=500)
Y = np.array(U(X, (A_v/rho0)**0.5, S)*10**-3)

# Scatter plot
#plt.scatter(up*10**-3, U_km, label='Data')

# Fitted curve plot
plt.plot(X*10**-3, Y, label='Fitted')

plt.xlabel('Particle velocity, km/s')
plt.ylabel('Shockwave velocity, km/s')
plt.title(f"Velocity Model Fit - {filename}")
plt.legend()  # Add a legend to differentiate between scatter and fitted curves

# Save the figure or display it
plt.savefig(f"{filename}_fit_velocity.png")
# Alternatively, you can use plt.show() to display the plot interactively
plt.show()


columns2 = ["P", "R"]
df_p = pd.read_excel(found_file, usecols=columns2)

x = np.array(df_p.R)
y = np.array(df_p.P)

data = np.column_stack((x, y))

if filename == "Corundum":
    data = data[data[:, 0] <= 6000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    #filter out high values that distort the fit
    
if filename == "Enstatite":
    data = data[data[:, 0] <= 4700]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    
if filename == "Forsterite":
    data = data[data[:, 1] <= 60000000000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    
if filename == "Grossular garnet":
    data = data[data[:, 1] <= 60000000000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    
if filename == "Gypsum":
    data = data[data[:, 0] >= 3300]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    #two sets
    
if filename == "Hematite":
    data = data[data[:, 0] <= 7000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
if filename == "Ilmenite":
    data = data[data[:, 0] <= 6300]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]

#if filename == "Kyanite":
 #   data = data[data[:, 0] <= 4750]
  #  truncated_x = data[:, 0]
   # truncated_y = data[:, 1]
    
if filename == "Magnetite":
    data = data[data[:, 1] <= 200000000000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    
if filename == "Olivine":
    data = data[data[:, 0] <= 5000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]   
    
if filename == "Orthoclase":
    data = data[data[:, 0] <= 4000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]    
    
elif filename =="Periclase":
    data = data[data[:,0] <= 5500]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]

elif filename =="Perovskite":
    data = data[data[:,0] <= 6000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    
elif filename =="Quartz, amorphous":
    data = data[data[:,0] >= 2400]
    data = data[data[:,0] <= 4400]
    data = data[data[:,1] <= 37000000000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]

elif filename =="Quartz, fused":
    data = data[data[:,0] <= 4300]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]    
    
elif filename =="Quartz":
    data = data[data[:,0] >= 2000]
    data = data[data[:, 1] <= 40*10**9]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]

elif filename =="Rutile":
    data = data[data[:,1] <= 150000000000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]
    
#elif filename =="Serpentine":
 #   data = data[data[:,1] <= 70000000000]
  #  truncated_x = data[:, 0]
   # truncated_y = data[:, 1]

elif filename =="Sillimanite":
    data = data[data[:,1] <= 57000000000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]

elif filename =="Tourmaline":
    data = data[data[:,0] <= 4500]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]    
    
elif filename =="Wollastonite":
    data = data[data[:,1] <= 80000000000]
    truncated_x = data[:, 0]
    truncated_y = data[:, 1]  
    
#elif filename =="Polycar":
 #   data = data[data[:,0] <= 5000]
  #  truncated_x = data[:, 0]
   # truncated_y = data[:, 1]        

else:
    truncated_x=data[:, 0]
    truncated_y=data[:, 1]
    
data = np.column_stack((truncated_x, truncated_y))


a = 0.5
Pdata = truncated_y
rho = truncated_x
A = A_v

def P(rho, b, B, E0):
    
    V0 = 1/rho0
    V=1/rho
    eta = (V0/V)
    mu = (eta-1)
    
    s = ((V0-V)/2)
    k = (E0 * eta**2)
    l = (A*mu + B*mu**2)
    
    aprime = (s/(V*k) * ((a*s)-V))
    bprime = (((a+b) * (s/V)) + (l*s/k) - 1)
    cprime = l
    
    root_term = bprime**2 - 4*aprime*cprime
    
    P_predicted = (-bprime - (root_term)**0.5) / (2*aprime)
    return P_predicted

initial_guess = [2, 2*A_v, 10E+6]
B_l = 0.25*A_v
B_u = 4*A_v
          
popt, pcov = scipy.optimize.curve_fit(P, rho, Pdata, p0 = initial_guess, bounds = ([0.2, B_l, 10E+6], [500, B_u, 2000E+6]))
[b_error, B_error, E0_error] = np.sqrt(np.diag(pcov))

b = popt[0]
B = popt[1]
E0 = popt[2]


print ('a =', 0.5)
print ('b =', round(b, 2), '+/-', round(b_error, 2))
print ('A =', round(A*10**-9, 2), '+/-', round(A_error, 2), 'GPa')
print ('B =', round(B*10**-9, 2), '+/-', round(B_error*10**-9, 2), 'GPa')
print ('E0 =', round(E0*10**-6, 2), '+/-', round(E0_error*10**-6, 2),'MJ/g')

print (rho0)

P_GPa = Pdata * 10**-9
predicted_values = P(rho, b, B, E0)*10**-9

residuals = abs(P_GPa - predicted_values)
squared_residuals = residuals**2
mean_squared_error = np.mean(squared_residuals)
RMSE = np.sqrt((mean_squared_error))/np.mean(P_GPa)

print('RMS =', round(RMSE, 2))

X = np.linspace(rho0, max(rho), num = 500)
Y = np.array(P(X, b, B, E0))*10**-9

#plt.scatter (rho, P_GPa, label ='Data')
plt.plot(X,Y, label = 'Fitted')

plt.xlabel('Density, kg/m3')
plt.ylabel('Pressure, GPa')
plt.title(f"Density Model Fit - {filename}")



plt.legend()



plt.savefig(f"{filename}_fit.png")


plt.show()

print (round(rho0, 2), "kg/m3")

if len(filename)<7:
    number = 7 - len(filename)
    outputname = filename + ('_' * number)
elif len(filename)>7:
    number = 7 - len(filename)
    outputname = filename[:number]
else:
    outputname = filename
    
print (outputname)


file_path = '/home/ksq20/iSALE/share/eos/{}.tillo'.format(outputname)
with open(file_path, "w") as file:
    file.write('#TILLO\n')
    file.write('--------------------------------------------------------------\n')
    file.write('--- Tillotson EOS parameter for *** {} ***\n'.format(filename))
    file.write('--------------------------------------------------------------\n')
    file.write('TL_RHO0               Reference density (kg/m^3)    : {}\n'.format(str('{:.2e}'.format(rho0)).replace('e','D')))
    file.write('TL_CHEAT              Spec. heat capacity (J/kg/K)  : 0.773D+03\n')
    file.write('TL_BULKA              Bulk modulus (Pa)             : {}\n'.format(str('{:.2e}'.format(A)).replace('e','D')))
    file.write('TL_BULKB              Tillotson B constant (Pa)     : {}\n'.format(str('{:.2e}'.format(B)).replace('e','D')))
    file.write('TL_EZERO              Tillotson E0 constant (J/kg)  : {}\n'.format(str('{:.2e}'.format(E0)).replace('e','D')))
    file.write('TL_THERMA             Tillotson a constant          : {}\n'.format(str('{:.2e}'.format(a)).replace('e','D')))
    file.write('TL_THERMB             Tillotson b constant          : {}\n'.format(str('{:.2e}'.format(b)).replace('e','D')))
    file.write('TL_ALPHA              Tillotson alpha constant      : 5.D0\n')
    file.write('TL_BETA               Tillotson beta constant       : 5.D0\n')
    file.write('TL_EIV                SIE incipient vaporisation    : 4.72D+6\n')
    file.write('TL_ECV                SIE complete vaporisation     : 18.2D+6\n')
    file.write('--------------------------------------------------------------\n')
    file.write('<<END')

