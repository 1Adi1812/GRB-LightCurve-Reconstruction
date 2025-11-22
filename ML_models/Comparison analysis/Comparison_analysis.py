import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import os
import csv
from lmfit import Model
import sympy as sym
import scipy.optimize as opt
from lmfit.models import LorentzianModel
from lmfit.models import GaussianModel
from lmfit import Model, Parameters, minimize, fit_report
from scipy.optimize import curve_fit
from scipy.stats import norm,cauchy,lognorm
from scipy import stats as st

#DEFINING WILLINGALE MODEL
def Willingale_if(t, F_a, alpha, T_a):
    if t<T_a:
        return F_a * np.exp(alpha - (t*alpha)/T_a)
    else:
        return F_a * np.power((t / T_a),(-alpha))

def Willingale(t, F_a, alpha, T_a):
    y = np.zeros(t.shape)
    for j in range(len(y)):
        y[j]=Willingale_if(t[j], F_a, alpha, T_a)
    return y

def log_Willingale_if(logt, logFa, alpha, logTa):
    if logt<logTa:
        return logFa + np.log10(np.e) * alpha * (1.0 - 10**logt/(10**logTa))
    else:
        return logFa - alpha * (logt - logTa)

def log_Willingale(logt, logFa, alpha, logTa):
    y = np.zeros(logt.shape)
    for j in range(len(y)):
        y[j]=log_Willingale_if(logt[j], logFa, alpha, logTa)
    return y

#DEFINING CAUCHY LORENTZIAN FUNCTION
def Cauchy_Lorentz(x, x_0, gamma):
    return ( 1 / (np.pi * gamma * (1 + ( (x-x_0) / gamma )**2 )))

header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']

GRB_parameters = pd.read_csv("/path/to/GRB/parameters/GRBs_parameters.csv", header=0, index_col=0)

GRB_new = pd.read_csv("/path/to/GRB/Names/GRB_NAMES.csv", header=0, usecols=[0])

#ARRAYS TO STORE VALUES OF ORIGINAL WILLINGALE PARAMETERS FOR ALL GRBs IN THE LOOP

log_T_a_val_all=[]
log_F_a_val_all=[]
alpha_val_all=[]

#ARRAYS TO STORE ORIGINAL WILLINGALE PARAMETER ERRORS FOR ALL GRBs IN THE LOOP

log_T_a_err_all=[]
log_F_a_err_all=[]
alpha_err_all=[]


#ARRAYS TO STORE THE WILLINGALE PARAMETERS AFTER RECONSTRUCTION FOR ALL GRBs IN THE LOOP

log_T_a_refit_val_all=[]
log_F_a_refit_val_all=[]
alpha_refit_val_all=[]

#ARRAYS TO STORE THE WILLINGALE PARAMETER ERRORS AFTER RECONSTRUCTION FOR ALL GRBs IN THE LOOP

log_T_a_refit_err_all=[]
log_F_a_refit_err_all=[]
alpha_refit_err_all=[]



#ARRAYS TO STORE VALUES OF ERROR FRACTIONS (BOTH ORIGINAL AND AFTER RECONSTRUCTION) FOR ALL GRBs IN THE LOOP

log_T_a_err_frac_all=[]
alpha_err_frac_all=[]
log_F_a_err_frac_all=[]

log_T_a_err_refit_frac_all=[]
alpha_err_refit_frac_all=[]
log_F_a_err_refit_frac_all=[]


#ARRAY TO STORE PERCENTAGE DECREASES IN ERRORS OF FITTING PARAMETERS FOR ALL GRBs IN THE LOOP

per_dec_log_F_a=[]
per_dec_alpha=[]
per_dec_log_T_a=[]

#ARRAY TO STORE GRB NAMES

Names=[]

#ARRAY TO STORE THE NUMBER OF OBSERVED DATAPOINTS

Obs_points=[]

#ARRAY TO STORE THE NUMBER OF OBSERVED DATAPOINTS IN THE PLATEAU

Obs_points_plat=[]
#ARRAY TO STORE LOG RESIDUALS

log_res=[]

for i in range(len(GRBIDs_arr)):
    GRB_Name = GRBIDs_arr[i]
    print(GRB_Name)
    trimmed_data = pd.read_csv("/path/to/your/original/GRB/data/"+GRB_Name+"_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)
    reconstructed_data = pd.read_csv("/path/to/your/reconstructed/data/"+GRB_Name+".csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)

    #DEFINING DENSITY FACTOR
    density_factor = 1
    #Here we obtain the fitting parameters.

    #Ta is in log scale. Fa in log scale. Alpha is linear scale.
    #And tt and tfinal in log scale.
    log_T_a = GRB_parameters.loc[GRB_Name, "logTa_best"]
    log_T_a_min = GRB_parameters.loc[GRB_Name, "logTa_min"]
    log_T_a_max = GRB_parameters.loc[GRB_Name, "logTa_max"]

    log_F_a = GRB_parameters.loc[GRB_Name, "logFa"]
    log_F_a_min = GRB_parameters.loc[GRB_Name, "logFa_min"]
    log_F_a_max = GRB_parameters.loc[GRB_Name, "logFa_max"]

    alpha = GRB_parameters.loc[GRB_Name, "alpha_best"]
    alpha_min = GRB_parameters.loc[GRB_Name, "alpha_min"]
    alpha_max = GRB_parameters.loc[GRB_Name, "alpha_max"]

    log_Tt = GRB_parameters.loc[GRB_Name, "logTt"]
    log_Tfinal = GRB_parameters.loc[GRB_Name, "logTfinal"]
    #FETCHING MAXIMUM AND MINIMUM VALUE OF FLUXES AND TIME VALUES FROM TRIMMED DATA
    #THESE VALUES ARE IN LINEAR SCALE
    max_fluxes = np.max(trimmed_data["flux"])
    min_fluxes = np.min(trimmed_data["flux"])

    max_ts = np.max(trimmed_data["t"])
    min_ts = np.min(trimmed_data["t"])


    #ABOVE VALUES IN LOG SCALE
    log_max_fluxes = np.log10(max_fluxes)
    log_min_fluxes = np.log10(min_fluxes)

    log_max_ts = np.log10(max_ts)
    log_min_ts = np.log10(min_ts)

    #DEFINING THE ERROR VALUES FROM DATA FILE IN LINEAR SCALE

    #for time
    positive_ts_err = trimmed_data["pos_t_err"]
    negative_ts_err = trimmed_data["neg_t_err"]

    #for flux
    positive_fluxes_err = trimmed_data["pos_flux_err"]
    negative_fluxes_err = trimmed_data["neg_flux_err"]


    #READING TIME AND FLUXES FROM THE TRIMMED DATA
    #THESE VALUES ARE IN LINEAR SCALE
    ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"].to_numpy()

    #ABOVE VALUES IN LOG SCALE
    log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)


    # ERROR ON THE FLUXES
    pos_fluxes= fluxes + positive_fluxes_err
    neg_fluxes= fluxes + negative_fluxes_err

    #SYMMETRIC ERROR VALUE
    fluxes_err_sym= (pos_fluxes - neg_fluxes )/2


    # GENERATES TIME VALUES AT EQUAL INTERVALS IN RANGE OF TS IN LINEAR SCALE
    # THIS IS TO BE USED FOR GENERATING THE TIMES AT WHICH WE RECONSTRUCT THE LC
    # IT IS EQUAL TO THE NUMBER OF DATA POINTS AS THE ORIGINAL LIGHT CURVE
    recon_t = np.geomspace(np.min(ts), np.max(ts), density_factor*len(ts))

    #ABOVE VALUE IN LOG SCALE
    log_recon_t = np.log10(recon_t)

    #CALCULATING ERRORBAR IN LINEAR SCALE
    ts_error = (positive_ts_err - negative_ts_err )/2
    fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

    #CALCULATING ERRORBAR IN LOG SCALE

    pos_log_fluxes = np.log10(pos_fluxes)
    neg_log_fluxes = np.log10(neg_fluxes)

    #ERROR PARAMETERS (LOG)
    log_F_a_err = (log_F_a_max - log_F_a_min)/2
    log_T_a_err= (log_T_a_max - log_T_a_min)/2
    alpha_err = (alpha_max - alpha_min)/2

    #CALLING LOG WILLINGALE FUNCTION ON LOG PARAMETER
    log_Willingale_line = log_Willingale(log_recon_t, log_F_a, alpha, log_T_a)

    #PLOTTING LOG WILLINGALE FIT USING LOG PARAMETERS AND FLUXES IN LOG SCALE
    plt.scatter(log_ts, log_fluxes)
    plt.plot(log_recon_t, log_Willingale_line, '-r', label="Observed Fit")
    plt.errorbar(log_ts, log_fluxes, linestyle='none', yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], marker='o', capsize=5, label="Trimmed Data")
    plt.xlabel("Time, t (s)")
    plt.ylabel("Flux")
    plt.title(GRB_Name)
    plt.legend()
    plt.show()

    #RESIDUALS IN LOG SCALE
    log_Willingale_line = log_Willingale(log_ts, log_F_a, alpha, log_T_a)
    #log_Willingale_min = log_Willingale(log_ts, log_F_a_min, alpha_min, log_T_a_min)  #THESE COMMENTED LINES CALCULATE THE MINIMUM AND MAXIMUM WILLINGALE FIT.
    #log_Willingale_max = log_Willingale(log_ts, log_F_a_max, alpha_max, log_T_a_max)



    ## GETTING ERROR DISTRIBUTION

    fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
    logfluxerrs = fluxes_error/(fluxes*np.log(10))

    log_F_a_err_frac=log_F_a_err/log_F_a
    alpha_err_frac= alpha_err/alpha
    log_T_a_err_frac=log_T_a_err/log_T_a

    if log_F_a_err_frac>=100:
      print(GRB_Name+" skipped")
      continue

    if log_T_a_err_frac>=100:
      print(GRB_Name+" skipped")
      continue

    if alpha_err_frac>=100:
      print(GRB_Name+" skipped")
      continue

    #DISPLAYING THE ORIGINAL ERROR FRACTIONS
    print("Error Fraction for original log Fa: "+str(log_F_a_err_frac))
    print("Error Fraction for original alpha: "+str(alpha_err_frac))
    print("Error Fraction for original log Ta: "+str(log_T_a_err_frac))

    func = log_Willingale
   # new_points = reconstructed_data["flux"]
   # new_points = np.log10(new_points)
   # new_time = reconstructed_data["t"]
   # new_time = np.log10(new_time)
   # xdata = np.append(new_time, log_ts)                   ## XDATA AND YDATA ARE NOW THE NEW TIME AND FLUX DATAPOINTS (ORIGINAL POINTS + RECONSTRUCTED POINTS).
   # ydata = np.append(new_points, log_fluxes)

    xdata = reconstructed_data["t"]
    ydata = reconstructed_data["flux"]

    # Filter out NaN values from xdata and ydata
    valid_indices = ~np.isnan(xdata) & ~np.isnan(ydata)
    xdata_cleaned = xdata[valid_indices]
    ydata_cleaned = ydata[valid_indices]
    xdata = np.log10(xdata_cleaned)
    ydata = np.log10(ydata_cleaned)

    xdata = xdata.values
    ydata = ydata.values

    plt.scatter(log_ts, log_fluxes, color='blue', label='Original Data',s=20,zorder=2)

    # Scatter plot for reconstructed data
    plt.scatter(xdata, ydata, color='orange', label='Reconstructed Data',s=50,zorder=1)

    # Add title and legend
    plt.title("Reconstructed")
    plt.legend()
    plt.show()




    params=Parameters() #DEFINITION OF WILLINGALE PARAMETERS AS A REQUIREMENT FOR THE USE OF LMFIT LIBRARY.

    params.add('log_F_a', value=log_F_a)
    params.add('alpha', value=alpha)
    params.add('log_T_a', value=log_T_a)


    #USING MINIMIZE FUNCTION FROM LMFIT LIBRARY TO DO THE REFITTING OF THE WILLINGALE MODEL ON RECONSTRUCTED GRB:

    def get_residual(params, x, y):           #RESIDUAL FUNCTION FOR USE IN LMFIT'S MINIMIZE FUNCTION.
      log_Ta=params['log_T_a'].value
      log_Fa=params['log_F_a'].value
      alph=params['alpha'].value
      model=log_Willingale(x, log_Fa, alph, log_Ta)

      return y-model

    #THE RESIDUAL FUNCTION, THE ORIGINAL WILLINGALE PARAMETERS, NEW X AND Y (TIME AND FLUX) DATA AS ARGUMENTS.
    result=minimize(get_residual, params, args=(xdata, ydata))

    print(fit_report(result)) #RESULTS OF THE REFITTING DISPLAYED USING fit_report FUNCTION IN LMFIT.

    # NEW VALUES OF THE WILLINGALE PARAMETERS
    log_F_a_refit = result.params['log_F_a'].value
    alpha_refit = result.params['alpha'].value
    log_T_a_refit = result.params['log_T_a'].value

    # NEW ERRORS ASSOCIATED WITH THE WILLINGALE PARAMETERS
    log_T_a_err_refit = result.params['log_T_a'].stderr
    log_F_a_err_refit = result.params['log_F_a'].stderr
    alpha_err_refit = result.params['alpha'].stderr

      #APPENDING THE ORIGINAL WILLINGALE PARAMETER VALUES AND THEIR ERRORS TO AN EMPTY ARRAY.
    #THIS WILL HAVE VALUES FOR EACH GRB
    log_T_a_val_all.append(log_T_a)
    log_F_a_val_all.append(log_F_a)
    alpha_val_all.append(alpha)

    log_T_a_err_all.append(log_T_a_err)
    log_F_a_err_all.append(log_F_a_err)
    alpha_err_all.append(alpha_err)

    #Multiple conditions if any parameter amounts to "None":
    try:
      log_T_a_refit_val_all.append(log_T_a_refit)
    except:
      log_T_a_refit_val_all.append(None)
    try:
      log_F_a_refit_val_all.append(log_F_a_refit)
    except:
      log_F_a_refit_val_all.append(None)
    try:
      alpha_refit_val_all.append(alpha_refit)
    except:
      alpha_refit_val_all.append(None)

    try:
      log_T_a_refit_err_all.append(log_T_a_err_refit)
    except:
      log_T_a_refit_err_all.append(None)
    try:
      log_F_a_refit_err_all.append(log_F_a_err_refit)
    except:
      log_F_a_refit_err_all.append(None)
    try:
      alpha_refit_err_all.append(alpha_err_refit)
    except:
      alpha_refit_err_all.append(None)


    try:
      log_F_a_err_refit_frac=log_F_a_err_refit/log_F_a_refit
    except:
      print('Cannot calculate log_F_a refit error fraction. The values of log_F_a after refit is: '+str(log_F_a_refit)+' and its associated error is: '+str(log_F_a_err_refit))
      log_F_a_err_refit_frac= None

    try:
      log_T_a_err_refit_frac=log_T_a_err_refit/log_T_a_refit
    except:
      print('Cannot calculate log_T_a refit error fraction. The values of log_T_a after refit is: '+str(log_T_a_refit)+' and its associated error is: '+str(log_T_a_err_refit))
      log_T_a_err_refit_frac= None

    try:
      alpha_err_refit_frac=alpha_err_refit/alpha_refit
    except:
      print('Cannot calculate alpha refit error fraction. The values of alpha after refit is: '+str(alpha_refit)+' and its associated error is: '+str(alpha_err_refit))
      alpha_err_refit_frac= None

    print("Error Fraction for reconstructed log Fa: "+str(log_F_a_err_refit_frac))
    print("Error Fraction for reconstructed alpha: "+str(alpha_err_refit_frac))
    print("Error Fraction for reconstructed log Ta: "+str(log_T_a_err_refit_frac))

    try:
       log_F_a_err_per_decrease=((abs(log_F_a_err_refit_frac)-abs(log_F_a_err_frac))/(abs(log_F_a_err_frac)))*100
    except:
      print("Cannot Calculate log_F_a_err_per_decrease as the refit error fraction is 'None'.")
      log_F_a_err_per_decrease=None

    try:
       log_T_a_err_per_decrease=((abs(log_T_a_err_refit_frac)-abs(log_T_a_err_frac))/(abs(log_T_a_err_frac)))*100
    except:
      print("Cannot Calculate log_T_a_err_per_decrease as the refit error fraction is 'None'.")
      log_T_a_err_per_decrease=None

    try:
       alpha_err_per_decrease=((abs(alpha_err_refit_frac)-abs(alpha_err_frac))/(abs(alpha_err_frac)))*100
    except:
      print("Cannot Calculate alpha_err_per_decrease as the refit error fraction is 'None'.")
      alpha_err_per_decrease=None


    #ADDING ORIGINAL ERROR FRACTION VALUES FOR ALL GRBs IN AN ARRAY
    log_T_a_err_frac_all.append(np.round(abs(log_T_a_err_frac), 6))
    alpha_err_frac_all.append(np.round(abs(alpha_err_frac), 6))
    log_F_a_err_frac_all.append(np.round(abs(log_F_a_err_frac), 6))


    #SAME APPENDING CODE FOR NEW ERROR FRACTIONS:
    try:
      log_T_a_err_refit_frac_all.append(np.round(abs(log_T_a_err_refit_frac), 6))
    except:
      log_T_a_err_refit_frac_all.append(None)

    try:
      alpha_err_refit_frac_all.append(np.round(abs(alpha_err_refit_frac),6))
    except:
      alpha_err_refit_frac_all.append(None)

    try:
      log_F_a_err_refit_frac_all.append(np.round(abs(log_F_a_err_refit_frac), 6))
    except:
      log_F_a_err_refit_frac_all.append(None)

    #ADDING PERCENTAGE DECREASE VALUES FOR ALL GRBs IN AN ARRAY:


    try:
      per_dec_log_F_a.append(np.round(log_F_a_err_per_decrease, 2))
    except:
      per_dec_log_F_a.append(None)

    try:
      per_dec_log_T_a.append(np.round(log_T_a_err_per_decrease, 2))
    except:
      per_dec_log_T_a.append(None)

    try:
      per_dec_alpha.append(np.round(alpha_err_per_decrease,2))
    except:
      per_dec_alpha.append(None)

    #APPENDING GRB NAMES IN AN ARRAY:

    Names.append(GRB_Name)

    #TOTAL NUMBER OF INITIAL DATAPOINTS FOR EACH GRB ARE STORED IN THIS ARRAY
    Obs_points.append(len(log_fluxes))



    #CODE TO FIND THE NUMBER OF DATAPOINTS ON THE PLATEAU FOR A GIVEN GRB

    ind=[] #array to store the indices of the times in log_ts where log_ts[i]<log_T_a
    plat_points=[]  #array to store the plateau points

    for i, elem in enumerate(log_ts):
      if elem<=log_T_a:
        ind.append(i)

    for i in range(0,len(ind)):
      plat_points.append(log_fluxes[ind[i]])

    Obs_points_plat.append(len(plat_points))



df=pd.DataFrame({'GRBID': Names , 'logTa': log_T_a_val_all, 'logFa': log_F_a_val_all, 'alpha': alpha_val_all, 'logTa_err': log_T_a_err_all, 'logFa_err': log_F_a_err_all, 'alpha_err': alpha_err_all, 'logTa_new': log_T_a_refit_val_all, 'logFa_new': log_F_a_refit_val_all, 'alpha_new': alpha_refit_val_all, 'logTa_err_new': log_T_a_refit_err_all, 'logFa_err_new': log_F_a_refit_err_all, 'alpha_err_new': alpha_refit_err_all, 'logTa_err_frac':log_T_a_err_frac_all, 'logFa_err_frac': log_F_a_err_frac_all, 'alpha_err_frac': alpha_err_frac_all, 'logTa_err_frac_recon': log_T_a_err_refit_frac_all, 'logFa_err_frac_recon': log_F_a_err_refit_frac_all , 'alpha_err_frac_recon': alpha_err_refit_frac_all, 'logTa_err_per_dec': per_dec_log_T_a ,'logFa_err_per_dec': per_dec_log_F_a, 'alpha_err_per_dec': per_dec_alpha})
df.head()

df.to_csv('/path/to/your/ouput/directory/Fourier_LCR.csv')


noise_level = 0.1   #For Willingale
df = pd.read_csv('/path/to/your/ouput/directory/Fourier_LCR.csv')
check=1000
print(noise_level)
plt.hist(df[df['logFa_err_per_dec']<=check]['logFa_err_per_dec'], bins=50)
plt.xlabel('Relative Percentage Decrease',fontsize=19)
plt.ylabel('Number of GRBs',fontsize=19)
plt.xlim(-100, 100)
plt.title('Fourier Transform Fitting ($\log(F_{\mathrm{a}})$)',fontsize=19)
plt.show()

plt.hist(df[df['logTa_err_per_dec']<=check]['logTa_err_per_dec'], bins=50)
plt.xlabel('Relative Percentage Decrease',fontsize=19)
plt.ylabel('Number of GRBs',fontsize=19)
plt.xlim(-100, 100)
plt.title('Fourier Transform Fitting ($\log(T_{\mathrm{a}})$)',fontsize=19)
plt.show()

plt.hist(df[df['alpha_err_per_dec']<=check]['alpha_err_per_dec'], bins=50)
plt.xlabel('Relative Percentage Decrease',fontsize=19)
plt.ylabel('Number of GRBs',fontsize=19)
plt.xlim(-100, 100)
plt.title('Fourier Transform Fitting (alpha)',fontsize=19)
plt.show()

print('total number of GRBs: '+str(df[df['logFa_err_per_dec']<=check]['logFa_err_per_dec'].count()))

decrease_count=df[df['logFa_err_per_dec']<=0]['logFa_err_per_dec'].count()
print('number of GRBs showing a decrease in error fraction of logFa: '+str(decrease_count))

total_count=df['logFa_err_per_dec'].count()
print('total number of GRBs for which table is generated: '+str(total_count))

print('percentage of GRBs showing a decrease in log Fa error fraction: '+str((decrease_count/total_count)*100)+'%')

print('number of GRBs showing an increase in error fraction (positive percentage): '+str(df[df['logFa_err_per_dec']>=0]['logFa_err_per_dec'].count()))

print('number of GRBs showing a percentage increase greater than '+str(check)+': '+str(df[df['logFa_err_per_dec']>=check]['logFa_err_per_dec'].count()))

check=100

print(df.query(f'logFa_err_per_dec < {check}')["logFa_err_per_dec"].mean())
print(df.query(f'logTa_err_per_dec < {check}')["logTa_err_per_dec"].mean())
print(df.query(f'alpha_err_per_dec < {check}')["alpha_err_per_dec"].mean())

