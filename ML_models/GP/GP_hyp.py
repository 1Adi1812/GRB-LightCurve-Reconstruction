import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import os
import csv
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import norm,cauchy,lognorm
from scipy import stats as st

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import Counter

from numpy.random import seed
from numpy.random import randn

header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']

GRB_Name='GRB170714A'

trimmed_data = pd.read_csv("/path/to/your/GRB/data/"+GRB_Name+"_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)

max_fluxes = np.max(trimmed_data["flux"])
min_fluxes = np.min(trimmed_data["flux"])

max_ts = np.max(trimmed_data["t"])
min_ts = np.min(trimmed_data["t"])


log_max_fluxes = np.log10(max_fluxes)
log_min_fluxes = np.log10(min_fluxes)

log_max_ts = np.log10(max_ts)
log_min_ts = np.log10(min_ts)

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
fluxes_err_sym= (pos_fluxes - neg_fluxes)/2

#CALCULATING ERRORBAR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

#CALCULATING ERRORBAR IN LOG SCALE
pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

#min gap 0.05, assign points based on the no. of original points
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

X_train, y_train = log_ts.reshape(-1,1), log_fluxes.reshape(-1,1)

# Original kernel from the LCR paper
kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))

#HYPERPARAMETER TUNING

# Create the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
'alpha': [1e-10, 1e-5, 1e-3, 1e-1, 1.0, 1.5, 2.0],
 'n_restarts_optimizer': [0, 5, 10, 20]
}

# Use GridSearchCV to search for the best hyperparameters
search = GridSearchCV(gp, param_grid, cv=10,n_jobs=-1)
search.fit(X_train, y_train)
best_params = search.best_params_

# Print the best parameters
print("Best parameters found: ", best_params)

best_alpha = best_params['alpha']
best_n = best_params['n_restarts_optimizer']

# Create and fit the GaussianProcessRegressor with the best alpha
gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=best_alpha, n_restarts_optimizer=best_n)
gaussian_process.fit(X_train, y_train)

# Predict using the fitted model
mean_prediction, std_prediction = gaussian_process.predict(log_recon_t, return_std=True)
mean_prediction = mean_prediction.ravel()

#CALCULATING TIME ERROR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_ts_error))

df=trimmed_data.copy(deep=True)

n_runs = 1 #100 -> we do not need the 100 loops here since it is already done 6 rows below; this 100 here is needed for overall analysis of reconstructed parameters
i = 0
while i<1: #in range(0, n_runs):
points=[]
for j in range(len(mean_prediction)):
  fitted_dist = st.norm(loc=mean_prediction[j], scale=std_prediction[j])
  point = np.random.choice(fitted_dist.rvs(size=100), size=1) - mean_prediction[j]
  points.append(point)

points = np.array(points).ravel()
new_points = mean_prediction + points

#df['recon_flux'] = new_points
i=i+1

## GETTING ERROR DISTRIBUTION

fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))


errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
#print("The parameters of the fit are: "+ str(errparameters))
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_errorbar=err_dist.rvs(size=len(log_recon_t))    # RECONSTRUCTED ERROR-BAR DEFINITION

# func = log_WillingaleGRB100725A
xdata = np.append(log_recon_t, log_ts)                   ## XDATA AND YDATA ARE NOW THE NEW TIME AND FLUX DATAPOINTS (ORIGINAL POINTS + RECONSTRUCTED POINTS).
ydata = np.append(new_points, log_fluxes)

plt.errorbar(log_recon_t, new_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow',zorder=3)
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], label=r"$\log_{10}\,flux$", linestyle="",zorder=4)
plt.plot(log_recon_t, mean_prediction, label="Mean prediction",zorder=2)
plt.scatter(X_train, y_train, marker=".", label="Observations",zorder=5)
plt.fill_between(
log_recon_t.ravel(),
mean_prediction - 1.96 * std_prediction,
mean_prediction + 1.96 * std_prediction,
alpha=0.5,
label=r"95% confidence interval",
zorder=1
)
plt.legend()
plt.xlabel("$time$")
plt.ylabel("$\log_{10}\,flux$")
_ = plt.title("GP regression on "+GRB_Name)

plt.savefig('/path/to/your/results/images/'+str(GRB_Name)+".png", dpi=300)
plt.show()

### EXPORT THE RECONSTRUCTED AND THE ORIGINAL DATA FRAMES IN A SINGLE COMBINED DATAFRAME ###

# print(len(log_recon_t))
# print(len(recon_logtimeerr))
# print(len(new_points))
# print(len(recon_errorbar))
data_frames = []

for k in range(0,len(log_recon_t)):
new_row={"t":10**(log_recon_t[k][0]), "pos_t_err":10**(recon_logtimeerr[k]), "neg_t_err":10**(recon_logtimeerr[k]), "flux":10**(new_points[k]), "pos_flux_err":10**(new_points[k])*np.log(10)*recon_errorbar[k], "neg_flux_err":10**(new_points[k])*np.log(10)*recon_errorbar[k]}
data_frames.append(pd.DataFrame([new_row]))

df = pd.concat(data_frames, ignore_index=True)
Names.append(GRB_Name)

df.to_csv('/path/to/your/results/csv/'+str(GRB_Name)+'.csv')

# 5 FOLD CROSS VALIDATION

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_mse_list = []
val_mse_list = []
fold = 1

for train_index, val_index in kf.split(log_ts):  
    print(f"Training fold {fold}...")

    #split Data
    X_train, X_val = log_ts[train_index].reshape(-1,1), log_ts[val_index].reshape(-1,1)
    y_train, y_val = log_fluxes[train_index].reshape(-1,1), log_fluxes[val_index].reshape(-1,1)

    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=best_alpha, n_restarts_optimizer=best_n)
    gaussian_process.fit(X_train, y_train)
    gaussian_process.kernel_

    #Prediction on training data
    train_pred, _ = gaussian_process.predict(X_train, return_std=True)
    train_pred=train_pred.ravel()
    train_mse = mean_squared_error(train_pred, y_train)
    train_mse_list.append(train_mse)
    
    #prediction on validation set
    val_pred, _ = gaussian_process.predict(X_val, return_std=True)
    val_pred= val_pred.ravel()
    val_mse = mean_squared_error(val_pred, y_val)
    val_mse_list.append(val_mse)
    
    fold+=1
    
# Calculate average MSE across folds
avg_train_mse = np.mean(train_mse_list)
avg_val_mse = np.mean(val_mse_list)

print(f"Average Train MSE: {avg_train_mse:.4f}")
print(f"Average Validation MSE: {avg_val_mse:.4f}")


file_path = '/path/to/your/results/MSE_results.csv'
write_header = not os.path.exists(file_path)
data = {
    "GRB_Name": [GRB_Name],
    "Train_MSE": [avg_train_mse],
    "Validation_MSE": [avg_val_mse],
}

df = pd.DataFrame(data)

df.to_csv(file_path, mode='a', header=write_header, index=False)
