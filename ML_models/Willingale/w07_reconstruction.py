
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import DataFrame as df
import os
import csv
from lmfit import Model
from lmfit import minimize, Parameters, fit_report
import sympy as sym
import scipy.optimize as opt
from lmfit.models import LorentzianModel
from lmfit.models import GaussianModel
from lmfit import Model
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

header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
GRB_parameters = pd.read_csv("/content/drive/MyDrive/545_GRBs_parameters.csv", header=0, index_col=0)

#DEFINING GRB NAME
GRB_Name = 'GRB210222B'
#FETCHING GRB DATA
trimmed_data = pd.read_csv("/path/to/your/GRB/data/"+GRB_Name+"_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)

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
ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"]

#ABOVE VALUES IN LOG SCALE
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

#ERROR ON THE FLUXES
pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err

#SYMMETRIC ERROR VALUE
fluxes_err_sym= ( pos_fluxes - neg_fluxes )/2

# min gap 0.05, assign points based on the no. of original points
min_gap = 0.05
recon_log_t = [log_ts[0]]
total_span = log_ts[-1] - log_ts[0]
if len(ts) > 500:   # densest LC
    fraction = 0.05
elif len(ts) > 250:
    fraction = 0.1
elif len(ts) > 100:
    fraction = 0.3
else:
    fraction = 0.4
n_points = max(20, int(fraction * len(ts)))
for i in range(len(ts) - 1):
    gap_size = log_ts[i+1] - log_ts[i]

    if gap_size > min_gap:
        # allocate points proportional to gap size
        interval_points = max(2, int(n_points * gap_size / total_span))
        interval = np.linspace(log_ts[i], log_ts[i+1], interval_points, endpoint=True)
        recon_log_t.extend(interval[1:])

recon_log_t = np.array(recon_log_t)
recon_t = 10**np.array(recon_log_t)
recon_t = np.unique(recon_t)

log_recon_t = np.log10(recon_t).reshape(-1, 1)

#CALCULATING ERRORBAR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2


#CALCULATING ERRORBAR IN LOG SCALE

pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

#ERROR PARAMETERS (LOG)
log_F_a_err = (log_F_a_max - log_F_a_min)/2
log_T_a_err=(log_T_a_max - log_T_a_min)/2
alpha_err = (alpha_max - alpha_min)/2

#RESIDUALS IN LOG SCALE

log_Willingale_line = log_Willingale(log_ts, log_F_a, alpha, log_T_a)

log_residuals = log_fluxes-log_Willingale_line  #EQUATION 2 IN THE MANUSCRIPT.

plt.figure(figsize=(8, 8))
plt.scatter(log_ts, log_residuals, color='red', marker='o')
plt.title('Log Residuals')
plt.show()



parameters = st.norm.fit(log_residuals, floc=0)
print("The parameters of the fit are: "+ str(parameters))
fitted_dist = st.norm(scale=parameters[1])


x = np.linspace(fitted_dist.ppf(0.001),fitted_dist.ppf(0.999), 100)
plt.plot(x,fitted_dist.pdf(x), 'k-', lw=2, label='fitted pdf')
plt.hist(log_residuals, density=True, histtype='stepfilled')
plt.xlim(-0.45,0.45)
plt.xlabel('$log_{10}$(Flux Residual)', fontsize="19")
plt.title(GRB_Name)
plt.show()

st.kstest(log_residuals, fitted_dist.cdf)

log_recon_t = np.ravel(log_recon_t)
log_Willingale_line = np.ravel(log_Willingale_line)

log_Willingale_line = log_Willingale(log_recon_t, log_F_a, alpha, log_T_a)

noise_level=0.1 #This can be changed to 10% OR 20% (0.1 and 0.2). Other values can also be chosen.
added_noise= 1 + noise_level

new_points= log_Willingale_line + added_noise * fitted_dist.rvs(size=len(log_recon_t))  #DEFINITION OF NEW POINTS (EQN. 4 IN THE MANUSCRIPT)

print("log_recon_t:", np.shape(log_recon_t))
print("log_Willingale_line:", np.shape(log_Willingale_line))
print("new_points:", np.shape(new_points))

plt.scatter(log_recon_t, new_points)
plt.plot(log_recon_t, log_Willingale_line)
plt.xlabel('$log_{10}(time)$ (s)',fontsize="19")
plt.ylabel("$log_{10}(Flux)$ ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="19")
plt.title(GRB_Name, fontsize="19")
plt.scatter(log_ts, log_fluxes)
plt.show()

## GETTING ERROR DISTRIBUTION

fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))

#CALCULATING TIME ERROR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t))

df=trimmed_data.copy(deep=True)

## PLOTTING ERROR-BAR DISTRIBUTION
plt.hist(logfluxerrs,density=True)
plt.title(GRB_Name)
plt.xlabel('$log_{10}$(Flux errors)', fontsize="19")
plt.show()

errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
print("The parameters of the fit are: "+ str(errparameters))
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

xs = np.linspace(err_dist.ppf(0.001),err_dist.ppf(0.999), 100)
plt.plot(xs, err_dist.pdf(xs), 'k-', lw=2, label='fitted pdf')
plt.hist(logfluxerrs, density=True, histtype='stepfilled')
plt.xlim(-0.45,0.45)
plt.xlabel('$log_{10}$(Flux Errors)', fontsize="19")
plt.title(GRB_Name)
plt.show()

recon_errorbar=np.abs(err_dist.rvs(size=len(log_recon_t)))    # RECONSTRUCTED FLUX ERROR-BAR DEFINITION

plt.errorbar(log_recon_t, new_points, yerr=recon_errorbar, color='orange' , marker='o', linestyle='None', capsize=5) #ERROR-BAR PLOT WITH RECONSTRUCTED ERROR-BARS.
plt.plot(log_recon_t, log_Willingale_line)
plt.xlabel('$log_{10}(time)$ (s)',fontsize="19")
plt.ylabel("$log_{10}(Flux)$ ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="19")
plt.rcParams["axes.edgecolor"]="black"
plt.rcParams["axes.linewidth"]=2.0
plt.title(str(GRB_Name)+' ('+str(noise_level*100 )+ '% noise level) ', fontsize="19")
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], color='tab:blue', marker='o', linestyle='None', capsize=5)

plt.savefig('/path/to/your/results/images/'+str(GRB_Name)+".png", dpi=300)
plt.show()

data_frames = []

for k in range(0,len(log_recon_t)):
    new_row={"t":10**(log_recon_t[k]), "pos_t_err":10**(recon_logtimeerr[k]), "neg_t_err":10**(recon_logtimeerr[k]), "flux":10**(new_points[k]), "pos_flux_err":10**(new_points[k])*np.log(10)*recon_errorbar[k], "neg_flux_err":10**(new_points[k])*np.log(10)*recon_errorbar[k]}
    data_frames.append(pd.DataFrame([new_row]))

df = pd.concat(data_frames, ignore_index=True)

df.to_csv('/path/to/your/results/csv/'+str(GRB_Name)+'.csv')


