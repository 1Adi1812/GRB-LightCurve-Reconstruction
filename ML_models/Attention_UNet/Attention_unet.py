# IMPORTS
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from scipy.optimize import curve_fit
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, ReLU, Dense
from tensorflow.keras.optimizers import Adam


GRB_Name = "GRB050820A"   #GRB you want to reconstruct

#Fixed parameters for the model
batch_size = 64
epochs = 1000

# PREPROCESSING

header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
trimmed_data = pd.read_csv(f"/path/to/your/GRB/data/{GRB_Name}_trimmed.csv", skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)
trimmed_data=trimmed_data.sort_values(by="t").reset_index(drop=True)

max_fluxes = np.max(trimmed_data["flux"])
min_fluxes = np.min(trimmed_data["flux"])

max_ts = np.max(trimmed_data["t"])
min_ts = np.min(trimmed_data["t"])

log_max_fluxes = np.log10(max_fluxes)
log_min_fluxes = np.log10(min_fluxes)

log_max_ts = np.log10(max_ts)
log_min_ts = np.log10(min_ts)

positive_ts_err = trimmed_data["pos_t_err"]
negative_ts_err = trimmed_data["neg_t_err"]

positive_fluxes_err = trimmed_data["pos_flux_err"]
negative_fluxes_err = trimmed_data["neg_flux_err"]

ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"].to_numpy()
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err

fluxes_err_sym= (pos_fluxes - neg_fluxes)/2

ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

#DEFINING THE GAPS IN THE LC
gaps = np.diff(log_ts)

#Min gap 0.05, assign points based on the no. of original points
min_gap = 0.05
recon_log_t = [log_ts[0]]
total_span = log_ts[-1] - log_ts[0]
if len(ts) > 500:
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
        interval_points = max(2, int(n_points * gap_size / total_span))
        interval = np.linspace(log_ts[i], log_ts[i+1], interval_points, endpoint=True)
        recon_log_t.extend(interval[1:])

recon_log_t = np.array(recon_log_t)
recon_t = 10**np.array(recon_log_t)
recon_t = np.unique(recon_t)

log_recon_t = np.log10(recon_t).reshape(-1, 1)


# Attention UNet Model

# Define Attention Block
def AttentionBlock1D(x, g, inter_channels):
    theta_x = Conv1D(inter_channels, kernel_size=1, strides=1, padding="same")(x)
    phi_g = Conv1D(inter_channels, kernel_size=1, strides=1, padding="same")(g)
    f = ReLU()(theta_x + phi_g)
    psi_f = Conv1D(1, kernel_size=1, strides=1, padding="same", activation="sigmoid")(f)
    return x * psi_f


def UNetWithAttention1D(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv1)
    pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)  # Ensure 'same' padding

    conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv2)
    pool2 = MaxPooling1D(pool_size=2, padding='same')(conv2)

    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv3)
    pool3 = MaxPooling1D(pool_size=2, padding='same')(conv3)

    # Bottleneck
    bottleneck = Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool3)
    bottleneck = Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(bottleneck)

    # Decoder
    upconv3 = UpSampling1D(size=2)(bottleneck)
    attention3 = AttentionBlock1D(conv3, upconv3, inter_channels=64)
    concat3 = concatenate([upconv3, attention3], axis=-1)
    conv_dec3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(concat3)
    conv_dec3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv_dec3)

    upconv2 = UpSampling1D(size=2)(conv_dec3)
    attention2 = AttentionBlock1D(conv2, upconv2, inter_channels=32)
    concat2 = concatenate([upconv2, attention2], axis=-1)
    conv_dec2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(concat2)
    conv_dec2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv_dec2)

    upconv1 = UpSampling1D(size=2)(conv_dec2)
    attention1 = AttentionBlock1D(conv1, upconv1, inter_channels=16)
    concat1 = concatenate([upconv1, attention1], axis=-1)
    conv_dec1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(concat1)
    conv_dec1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv_dec1)

    outputs = Conv1D(1, kernel_size=1, activation=None)(conv_dec1)
    outputs = Flatten()(outputs)
    outputs = Dense(input_shape[-1], activation="linear")(outputs)

    model = Model(inputs, outputs)
    return model



input_shape = (1, 1)

model = UNetWithAttention1D(input_shape=(1, 1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Training
X_train = log_ts.reshape(-1, 1, 1)
y_train = log_fluxes.reshape(-1, 1, 1)
model.fit(X_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size)

# Predictions
x_test = log_recon_t.reshape(-1, 1, 1)
log_recon_fx = model.predict(x_test).flatten()
print(log_recon_fx.shape, log_recon_t.shape)
    
recon_fluxes_up = model.predict(x_test).flatten()  #Model Predictions

#ADDING NOISE 

ts_error = (positive_ts_err - negative_ts_err)/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t)) # len(log_ts_error)
fluxes_error = (trimmed_data["pos_flux_err"] - trimmed_data["neg_flux_err"]) / 2
logfluxerrs = fluxes_error / (fluxes * np.log(10))
errparameters = st.norm.fit(logfluxerrs)
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

# Generate recon_errorbar with the same length as log_recon_t
recon_errorbar = err_dist.rvs(size=len(recon_fluxes_up))
recon_errorbar = np.where(recon_errorbar < 0, 0, recon_errorbar)

# Generate point-specific noise for each prediction
point_specific_noise = [
    st.norm(loc=pred, scale=err).rvs() - pred
    for pred, err in zip(recon_fluxes_up, recon_errorbar)
]

# Convert point_specific_noise to a numpy array
point_specific_noise = np.array(point_specific_noise)

# Add noise to predictions
jiggled_points = recon_fluxes_up + point_specific_noise

# Downsample recon_fluxes_up and jiggled_points
factor = len(recon_fluxes_up) // len(log_recon_t)
recon_fluxes_up_downsampled = recon_fluxes_up[::factor]
jiggled_points_downsampled = jiggled_points[::factor]
recon_errorbar_downsampled = recon_errorbar[::factor]

# Ensure the lengths match
if len(recon_fluxes_up_downsampled) != len(log_recon_t):
    raise ValueError(f"Downsampled recon_fluxes_up ({len(recon_fluxes_up_downsampled)}) does not match log_recon_t ({len(log_recon_t)}).")

# Debugging print statements (optional)
print(f"log_recon_t length: {len(log_recon_t)}")
print(f"recon_fluxes_up length: {len(recon_fluxes_up)}")
print(f"point_specific_noise length: {len(point_specific_noise)}")

num_samples = 1000

random_samples = np.array([
    st.norm(loc=0, scale=err).rvs(num_samples)
    for err in recon_errorbar_downsampled
]).T

jiggled_realizations = recon_fluxes_up_downsampled + random_samples

ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)

log_recon_t = log_recon_t.flatten()
ci_95_lower = ci_95_lower.flatten()
ci_95_upper = ci_95_upper.flatten()

# Plotting the final reconstructed image
print("SAVING PLOT...\n")
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes - np.log10(fluxes - fluxes_error), np.log10(fluxes + fluxes_error) - log_fluxes], linestyle="", zorder=4)  
plt.errorbar(log_recon_t, jiggled_points_downsampled, yerr=np.abs(recon_errorbar_downsampled), marker='o', capsize=5, color='yellow', label='Reconstructed Points', zorder=3)   
plt.scatter(log_ts, log_fluxes, label='Observed Points', color='blue', zorder=5)  
plt.plot(log_recon_t, recon_fluxes_up_downsampled, label='Mean predictions', linewidth=2, color='red', zorder=2)  
plt.fill_between(log_recon_t, ci_95_lower, ci_95_upper,color='orange', alpha=0.5, label='95% Confidence Interval',zorder=1)
plt.legend(loc='lower left')
plt.xlabel('log$_{10}$(Time) (s)',fontsize="15")
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="15")
plt.title(f'Attention UNet on '+str(GRB_Name), fontsize = "18")
plt.savefig('/path/to/your/result/directory/'+str(GRB_Name)+".png", dpi=300)
plt.show()


# Save as CSV
print("SAVING DATAFRAME...")

df = trimmed_data.copy()
for k, log_t in enumerate(log_recon_t.flatten()):
    new_row = {
        "t": 10**log_t,
        "pos_t_err": 10**recon_errorbar_downsampled[k],
        "neg_t_err": 10**recon_errorbar_downsampled[k],
        "flux": 10**jiggled_points_downsampled[k],
        "pos_flux_err": 10**jiggled_points_downsampled[k] * np.log(10) * recon_errorbar_downsampled[k],
        "neg_flux_err": 10**jiggled_points_downsampled[k] * np.log(10) * recon_errorbar_downsampled[k]
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv('/path/to/your/result/directory/'+str(GRB_Name)+'.csv')


#5 FOLD CROSS VALIDATION

kf = KFold(n_splits=5)

train_mse, test_mse = [], []
pbar = tqdm(5)
for i, (train_index, test_index) in enumerate(kf.split(log_ts, log_fluxes)):
    input_shape = (1, 1)
    model = UNetWithAttention1D(input_shape=(1, 1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Training
    X_train = log_ts[train_index, ...].reshape(-1, 1)
    y_train = log_fluxes[train_index, ...].reshape(-1, 1)
    X_test = log_ts[test_index].reshape(-1, 1)
    y_test = log_fluxes[test_index].reshape(-1, 1)

    model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # print(y_train.shape, train_pred.shape)
    # print(y_test.shape, test_pred.shape)

    train_mse.append(mean_squared_error(y_train, train_pred))
    test_mse.append(mean_squared_error(y_test, test_pred))
    pbar.update(1)

np.save(f"/path/to/your/results/{GRB_Name}/train_mse.npy", np.array(train_mse))
np.save(f"/path/to/your/results/{GRB_Name}/test_mse.npy", np.array(test_mse))


