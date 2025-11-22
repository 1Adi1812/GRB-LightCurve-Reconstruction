
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as st
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import initializers
import os

seed_value = 42 #Setting the seed 
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

GRB_Name = "GRB050820A"  #GRB you want to reconstruct

final_train_mse = []
final_val_mse = []
final_train_z =[]
final_val_z =[]

header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
trimmed_data = pd.read_csv("/path/to/your/GRB/data/", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)

max_fluxes = np.max(trimmed_data["flux"])
min_fluxes = np.min(trimmed_data["flux"])

max_ts = np.max(trimmed_data["t"])
min_ts = np.min(trimmed_data["t"])

#CONVERTING TO LOG SCALE
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
ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"].to_numpy()

log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

# ERROR ON THE FLUXES
pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err


#DEFINING THE GAPS IN THE LC
gaps = np.diff(log_ts)

#Min gap 0.05, assign points based on the no. of original points
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

#RECONSTRUCTION Bi-LSTM CODE

X = np.array(log_ts).reshape(-1, 1)
y = np.array(log_fluxes).reshape(-1, 1)

# Initialize the scaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X= X_scaled
y= y_scaled

he_init = initializers.HeNormal()

X_seq = X_scaled.reshape(X_scaled.shape[0], 1, 1)

model = Sequential()

model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)))
model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)))
model.add(Bidirectional(LSTM(100, kernel_initializer=he_init)))

#dense layer with leaky ReLU activation for non linear data
model.add(Dense(1, activation='leaky_relu'))

model.compile(optimizer='adam', loss='mean_squared_error')

#Fit the model
history = model.fit(X_seq, y,
                    epochs=100,
                    batch_size=3,
                    verbose=1)


recon_scaled = scaler_X.fit_transform(log_recon_t)

log_recon_seq = recon_scaled.reshape(recon_scaled.shape[0], 1, 1)

#Predict on the gaps
recon_pred = model.predict(log_recon_seq)

predictions = scaler_y.inverse_transform(recon_pred)  #inverse transformation

#Plot the predictions of the model
plt.scatter(log_ts,log_fluxes,label='actual')
plt.scatter(log_recon_t,predictions,label='predicted')

plt.xlabel('Time')
plt.ylabel('Flux')
plt.title(f'Predicted vs Actual Light Curves')
plt.legend()
plt.show()

#ADDING NOISE

#CALCULATING TIME ERROR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err)/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t)) # len(log_ts_error)

fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))

errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_errorbar=err_dist.rvs(size=len(log_recon_t))

#Point specific noise
point_specific_noise = []
for j in range(len(predictions)):
    fitted_dist = norm(loc=predictions[j], scale=recon_errorbar[j])
    point_noise = fitted_dist.rvs() - predictions[j]
    point_specific_noise.append(point_noise)

point_specific_noise = np.array(point_specific_noise)

#Jiggle reconstructed points
jiggled_points = predictions + point_specific_noise

num_samples = 1000  # Number of realizations
jiggled_realizations = []

for _ in range(num_samples):
    point_specific_noise = []
    for j in range(len(predictions)):
        fitted_dist = norm(loc=predictions[j], scale=recon_errorbar[j])
        point_noise = fitted_dist.rvs() - predictions[j]
        point_specific_noise.append(point_noise)
    jiggled_realizations.append(predictions + np.array(point_specific_noise))

jiggled_realizations = np.array(jiggled_realizations)

# Compute mean and 95% confidence intervals
mean_jiggled = np.mean(jiggled_realizations, axis=0)
ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)  # 2.5th percentile
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)  # 97.5th percentile

ci_95_lower = ci_95_lower.flatten()
ci_95_upper = ci_95_upper.flatten()
log_recon_t = log_recon_t.flatten()
jiggled_points = jiggled_points.flatten()

#PLOTTING FINAL RECONSTRUCTION
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], linestyle="",zorder=4)
plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, label = 'Recontructed Points', color='yellow',zorder=3)

plt.scatter(log_ts,log_fluxes,label='Observed Points',zorder=5)
plt.plot(log_recon_t,predictions,label='Mean predictions',zorder=2)
plt.fill_between(log_recon_t, ci_95_lower, ci_95_upper,color='orange', alpha=0.5, label='95% Confidence Interval',zorder=1)
plt.legend(loc='lower left')
plt.xlabel('log$_{10}$(Time) (s)',fontsize="15")
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="15")
plt.title(f'LSTM on '+str(GRB_Name), fontsize = "18")
plt.savefig('/path/to/your/result/directory/'+str(GRB_Name)+".png", dpi=300)
plt.show()

#SAVING CSV FILES OF RECONSTRUCTED DATA
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

df.to_csv('/path/to/your/result/directory/'+str(GRB_Name)+'.csv')

# 5 FOLD CROSS VALIDATION

kf = KFold(n_splits=5, shuffle=True, random_state=42)

X = np.array(log_ts).reshape(-1, 1)
y = np.array(log_fluxes).reshape(-1, 1)

# Initialize the scaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X= X_scaled
y= y_scaled

fold = 1
f_train_mse = []
f_val_mse = []
train_z_loss_list = []
val_z_loss_list = []

for train_index, val_index in kf.split(X):
    print(f"Training fold {fold}...")

    # Split data
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    X_train = X_train.reshape((X_train.shape[0], 1, 1))  # (samples, timesteps, features) only changing it to 3D. no scaling over here
    X_val = X_val.reshape((X_val.shape[0], 1, 1))

    he_init = initializers.HeNormal()

    model = Sequential()

    model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
    model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, kernel_initializer=he_init)))

    #dense layer with leaky ReLU activation for non linear data
    model.add(Dense(1, activation='leaky_relu'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=3,
                        validation_data=(X_val, y_val),
                        verbose=1)

    print(f"Fold {fold} - Training MSE: {train_mse}, Validation MSE: {val_mse}")

    # Predict on training and validation sets
    y_train_pred = model.predict(X_train).flatten()
    y_val_pred = model.predict(X_val).flatten()

    #inverse transformation on predictions
    train_inv = scaler_y.inverse_transform((y_train).reshape(-1,1)).flatten()
    train_inv_pred = scaler_y.inverse_transform((y_train_pred).reshape(-1,1)).flatten()

    val_inv = scaler_y.inverse_transform((y_val).reshape(-1,1)).flatten()
    val_inv_pred = scaler_y.inverse_transform((y_val_pred).reshape(-1,1)).flatten()

    #calculating MSE
    train_mse = np.mean((train_inv - train_inv_pred) ** 2)
    val_mse = np.mean((val_inv - val_inv_pred) ** 2)

    f_train_mse.append(train_mse)
    f_val_mse.append(val_mse)

    fold += 1

print("5 fold train mse", f_train_mse)
print("5 fold val_mse",f_val_mse)
mean_train_mse = np.mean(f_train_mse)
mean_val_mse = np.mean(f_val_mse)

print(f"Mean Training MSE across folds: {mean_train_mse}")
print(f"Mean Validation MSE across folds: {mean_val_mse}")

#SAVING 5 FOLD CV RESULTS
file_path = f"/path/to/your/result/directory/{GRB_Name}_MSE.csv"
write_header = not os.path.exists(file_path)  # Only write the header if the file doesn't exist

data = {
    "GRB Name": [GRB_Name],
    "Final Train MSE": [mean_train_mse],
    "Final Validation MSE":[mean_val_mse]
}

df = pd.DataFrame(data)

df.to_csv(file_path, header=write_header, index=False)
