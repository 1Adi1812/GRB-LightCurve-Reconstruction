#IMPORTS
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import os
import csv
from tqdm import tqdm
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import norm,cauchy,lognorm
from scipy import stats as st
from sklearn import preprocessing
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from collections import Counter
from numpy.random import seed
from numpy.random import randn


GRB_Name = "GRB050820A"

header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
trimmed_data = pd.read_csv("/path/to/your/GRB/Data/"+GRB_Name+"_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)


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

# Defining gaps for reconstruction
gaps = np.diff(log_ts)

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

#CALCULATING ERRORBAR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

#CALCULATING ERRORBAR IN LOG SCALE
pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

log_fluxes = np.log10(fluxes).reshape(-1,1)
log_ts = np.log10(ts).reshape(-1,1)

scaler_flux = MinMaxScaler()
scaler_ts = MinMaxScaler()

# Fit and transform log_fluxes and log_ts
log_fluxes = scaler_flux.fit_transform(log_fluxes)
log_ts = scaler_ts.fit_transform(log_ts)

kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))

# Fit GP with best kernel
gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
gp.fit(log_ts, log_fluxes)

# Get GP predictions
y_train_gp = gp.predict(log_ts).reshape(-1, 1)

# Calculate residuals
residuals_train = log_fluxes - y_train_gp

# Fine-tune Random Forest with GridSearchCV
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# Use RandomizedSearchCV instead of GridSearchCV
rf = RandomForestRegressor( n_estimators=100)
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings sampled
    cv=3,       # Reduced number of cross-validation folds
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

random_search.fit(log_fluxes, residuals_train)

# Get best RF model
best_rf = random_search.best_estimator_

# Predict residuals with best RF model
predictions = best_rf.predict(log_ts).reshape(-1, 1)

# Combine GP and RF predictions
final_pred = predictions + y_train_gp

# Inverse transform predictions to original scale
final_pred = scaler_flux.inverse_transform(final_pred)
log_fluxes = scaler_flux.inverse_transform(log_fluxes)
log_ts = scaler_ts.inverse_transform(log_ts)

# Calculate MSE and R² on original scale
r2 = r2_score(log_fluxes, final_pred)
mse = mean_squared_error(log_fluxes, final_pred)

print(f"R² Score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")


print(f"log_recon_seq shape: {log_recon_t.shape}")
log_recon_ts = scaler_ts.transform(log_recon_t)

#Predict on new time
gp_preds = gp.predict(log_recon_ts).reshape(-1,1)
recon_pred = best_rf.predict(log_recon_ts).reshape(-1,1)

final_pred = gp_preds+recon_pred

final_preds = scaler_flux.inverse_transform(final_pred)


#CALCULATING TIME ERROR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err)/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t)) # len(log_ts_error)

# ADDING NOISE
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))

errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_errorbar=err_dist.rvs(size=len(log_recon_t))

#Point specific noise
point_specific_noise = []
for j in range(len(final_preds)):
    fitted_dist = st.norm(loc=final_preds[j], scale=recon_errorbar[j])
    point_noise = fitted_dist.rvs() - final_preds[j]
    point_specific_noise.append(point_noise)
point_specific_noise = np.array(point_specific_noise)

#Jiggle reconstructed points
jiggled_points = final_preds + point_specific_noise

# Generate multiple realizations with noise
num_samples = 1000
jiggled_realizations = []

for _ in range(num_samples):
    point_specific_noise = []
    for j in range(len(final_preds)):
        fitted_dist = norm(loc=final_preds[j], scale=recon_errorbar[j])
        point_noise = fitted_dist.rvs() - final_preds[j]
        point_specific_noise.append(point_noise)
    jiggled_realizations.append(final_preds + np.array(point_specific_noise))

jiggled_realizations = np.array(jiggled_realizations)

# Calculate statistics
mean_jiggled = np.mean(jiggled_realizations, axis=0)
ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)

log_fluxes = log_fluxes.flatten()
log_ts = log_ts.flatten()


# Flatten arrays for plotting
ci_95_lower = ci_95_lower.flatten()
ci_95_upper = ci_95_upper.flatten()
log_recon_t = log_recon_t.flatten()
jiggled_points = (final_preds + point_specific_noise).flatten()

# Create final plot
#plt.figure(figsize=(10, 6))
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], linestyle="", zorder=4)
plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow',label = "Reconstructed Points", zorder=3)

plt.scatter(log_ts, log_fluxes, label='Observed Points', zorder=5)
plt.plot(log_recon_t, final_preds, label='Mean predictions', zorder=2)
plt.fill_between(log_recon_t, np.squeeze(ci_95_lower), np.squeeze(ci_95_upper), color='orange', alpha=0.5, label='95% Confidence Interval', zorder=1)

plt.legend(loc='lower left', fontsize = "10")
plt.xlabel('log$_{10}$(Time) (s)',fontsize="15")
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="15")
plt.title(f'GP-RF on {GRB_Name}', fontsize=18)

plt.savefig(f'/path/to/your/results/{GRB_Name}.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results to CSV
df = trimmed_data.copy(deep=True)
for k in range(0, len(log_recon_t)):
    new_row = {
        "t": 10**log_recon_t[k],
        "pos_t_err": 10**recon_logtimeerr[k],
        "neg_t_err": 10**recon_logtimeerr[k],
        "flux": 10**jiggled_points[k],
        "pos_flux_err": 10**jiggled_points[k] * np.log(10) * recon_errorbar[k],
        "neg_flux_err": 10**jiggled_points[k] * np.log(10) * recon_errorbar[k]
    }

    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
df.to_csv(f'/path/to/your/results/{GRB_Name}.csv')

# 5 FOLD CROSS VALIDATION

kf = KFold(n_splits=5)

# Store losses
kf_train_mse_losses = []
kf_test_mse_losses = []

# Perform K-Fold Cross-Validation
for i, (train_index, test_index) in enumerate(kf.split(log_ts, log_fluxes)):
    print(f"Cross Validation Split: {i + 1}")

    # Create training and testing datasets
    train_ts, train_fluxes = log_ts[train_index], log_fluxes[train_index]
    test_ts, test_fluxes = log_ts[test_index], log_fluxes[test_index]

    # Reshape inputs to 2D (necessary for many ML models)
    train_ts = train_ts.reshape(-1, 1)
    test_ts = test_ts.reshape(-1, 1)
    #Reshape train_fluxes and test_fluxes to 2D
    train_fluxes = train_fluxes.reshape(-1, 1)
    test_fluxes = test_fluxes.reshape(-1, 1)


    #min max scaling
    scaler_flux = MinMaxScaler()
    scaler_ts = MinMaxScaler()

    train_ts = scaler_ts.fit_transform(train_ts)
    test_ts = scaler_ts.transform(test_ts)

    train_fluxes = scaler_flux.fit_transform(train_fluxes)
    test_fluxes = scaler_flux.transform(test_fluxes)

    # Train the GP+RF model
    param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]}

    kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
    gp.fit(train_ts, train_fluxes)

    gp_preds = gp.predict(train_ts).reshape(-1, 1)

    # Calculate residuals
    residuals_train = train_fluxes - gp_preds

    random_search.fit(train_ts, residuals_train)
    best_rf = random_search.best_estimator_
    rf_preds = best_rf.predict(train_ts).reshape(-1,1)

    train_preds = gp_preds + rf_preds

    # Evaluate on testing set
    gp_test_preds = gp.predict(test_ts).reshape(-1, 1)
    rf_test_preds = best_rf.predict(test_ts).reshape(-1,1)

    test_preds = gp_test_preds + rf_test_preds

    #inverse scaling
    train_fluxes = scaler_flux.inverse_transform(train_fluxes)
    test_fluxes = scaler_flux.inverse_transform(test_fluxes)
    train_ts = scaler_ts.inverse_transform(train_ts)
    test_ts = scaler_ts.inverse_transform(test_ts)
    train_preds = scaler_flux.inverse_transform(train_preds)
    test_preds = scaler_flux.inverse_transform(test_preds)

    #mse predictions
    train_mse_loss = mean_squared_error(train_fluxes, train_preds)

    test_mse_loss = mean_squared_error(test_fluxes, test_preds)

    print(f"TRAINING - MSE: {train_mse_loss:.4f}")
    print(f"TESTING - MSE: {test_mse_loss:.4f}")

    # Store results
    kf_train_mse_losses.append(train_mse_loss)

    kf_test_mse_losses.append(test_mse_loss)


# Compute average test loss
avg_test_loss = np.mean(kf_test_mse_losses)
avg_train_loss = np.mean(kf_train_mse_losses)

print(f"Final Training Loss: {round(avg_train_loss, 4)}")
print(f"Final Test Loss: {round(avg_test_loss, 4)}")

# Reshape loss metrics for each fold
kf_train_mse_losses = np.array(kf_train_mse_losses)
kf_test_mse_losses = np.array(kf_test_mse_losses)

df1 = pd.DataFrame(kf_train_mse_losses)
df2 = pd.DataFrame(kf_test_mse_losses)

df1.to_csv('/path/to/your/results/'+str(GRB_Name)+'_train_mse.csv')
df2.to_csv('/path/to/your/results/'+str(GRB_Name)+'_test_mse.csv')



