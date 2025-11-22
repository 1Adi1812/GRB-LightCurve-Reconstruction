import numpy as np
import pandas as pd
from google.colab import drive
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import stats as st
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

folder_path='path to the GRB file'
grb_name='GRB Name'
trimmed_data=pd.read_csv(folder_path+grb_name+'.csv')
GRB_Name=grb_name

density_factor = 1
trim_t_val = trimmed_data["t"] if "t" in trimmed_data else trimmed_data["time_sec"]
ts, fluxes = trim_t_val.to_numpy(), trimmed_data["flux"].to_numpy()
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

positive_fluxes_err = trimmed_data["pos_flux_err"] if "pos_flux_err" in trimmed_data else trimmed_data["flux_errPos"]
negative_fluxes_err = trimmed_data["neg_flux_err"] if "neg_flux_err" in trimmed_data else trimmed_data["flux_errNeg"]
pos_fluxes = fluxes + positive_fluxes_err
neg_fluxes = fluxes + negative_fluxes_err
pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

# Normalize data
log_ts_mean = np.mean(log_ts)
log_ts_std = np.std(log_ts)
log_ts_norm = (log_ts - log_ts_mean) / log_ts_std
log_fluxes_mean = np.mean(log_fluxes)
log_fluxes_std = np.std(log_fluxes)
log_fluxes_norm = (log_fluxes - log_fluxes_mean) / log_fluxes_std

# Prepare data
train_x = log_ts_norm.reshape(-1, 1)
train_y = log_fluxes_norm.reshape(-1, 1)

# Split into train/validation for monitoring (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Build model
lr = 0.001
adam = Adam(learning_rate=lr)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=1, kernel_initializer=tf.keras.initializers.HeUniform()))
model.add(Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()))
model.add(Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()))
model.add(Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()))
model.add(Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()))
model.add(Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()))
model.add(Dense(1))

model.compile(loss='mse', optimizer=adam)

# Training parameters
epochs = 5000
batch_size = 24
patience = 50
patience_counter = 0
best_val_loss = float("inf")
threshold = 0.1
times = 5

epoch = 0
while epoch < epochs:
    epoch += 1

    # Train for 1 epoch at a time
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

    # Compute validation loss
    val_preds = model.predict(X_val, batch_size=batch_size, verbose=0)
    val_mse_loss = np.mean((y_val - val_preds) ** 2)

    if round(val_mse_loss, 4) < best_val_loss:
        best_val_loss = round(val_mse_loss, 4)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        if val_mse_loss > threshold and times != 0:
            epochs += patience  # extend training
            times -= 1
            patience_counter = 0
        else:
            print(f"Stopped at epoch {epoch} with best validation MSE = {best_val_loss}")
            break

#GAP-AWARE reconstruction
gaps = np.diff(log_ts)

#new code for min gap 0.05, assign points based on the no. of original points
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
test_x = (log_recon_t - log_ts_mean) / log_ts_std

# Predict mean
mean_prediction = model.predict(test_x).flatten()
mean_prediction_denorm = mean_prediction * log_fluxes_std + log_fluxes_mean

# Compute best fit distribution for noise
fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2
logfluxerrs = fluxes_error / (fluxes * np.log(10))
distributions = [st.norm, st.laplace]
fits = {}
for dist in distributions:
    params = dist.fit(logfluxerrs)
    loglikelihood = np.sum(dist.logpdf(logfluxerrs, *params))
    fits[dist.name] = (params, loglikelihood)
best_dist_name = max(fits, key=lambda d: fits[d][1])
best_params = fits[best_dist_name][0]
best_dist = getattr(st, best_dist_name)

points = []
for j in range(len(mean_prediction)):
    noise = 3.5 * (best_dist.rvs(*best_params, size=1)[0] - best_params[0])
    points.append(noise)
points = np.array(points)
new_points = mean_prediction + points
log_reconstructed_flux = (new_points * log_fluxes_std) + log_fluxes_mean
# log_reconstructed_flux = (mean_prediction * log_fluxes_std) + log_fluxes_mean

# Generate error bars for reconstructed points
recon_errorbar = []
for _ in range(len(log_recon_t)):
    err_sample = best_dist.rvs(*best_params, size=1)[0]
    recon_errorbar.append(err_sample)
recon_errorbar = np.abs(np.array(recon_errorbar))

# Compute confidence interval
train_pred = model.predict(train_x).flatten()
residuals = train_y - train_pred
sigma = np.std(residuals)  # Standard deviation in normalized units
std_y_denorm = sigma * log_fluxes_std  # Denormalized to log10(flux) units
lower_denorm = mean_prediction_denorm - 2 * std_y_denorm
upper_denorm = mean_prediction_denorm + 2 * std_y_denorm

# Define denormalized variables for plotting
train_x_denorm = log_ts
train_y_denorm = log_fluxes
test_x_denorm = log_recon_t.flatten()

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes - neg_log_fluxes, pos_log_fluxes - log_fluxes], linestyle="", label="Original Data")
plt.errorbar(test_x_denorm, log_reconstructed_flux, yerr=recon_errorbar, linestyle='none', marker='o', capsize=5, color='yellow', label="Reconstructed Points")
plt.scatter(log_ts, log_fluxes, label='Observed Points')
plt.plot(test_x_denorm, mean_prediction_denorm, label='Mean Prediction')
plt.fill_between(test_x_denorm, lower_denorm, upper_denorm, alpha=0.5, color='orange', label='95% Confidence Region')
plt.legend(loc='lower left')
plt.xlabel('log$_{10}$(Time) (s)', fontsize=15)
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)", fontsize=15)
plt.title(f'MLP on {GRB_Name}', fontsize=18)
plt.savefig(folder_path+f"{GRB_Name}.png", dpi=300)
print("\n-----RECONSTRUCTED GRB-----\n")
plt.show()

# Save reconstructed data
ts_error = (trimmed_data["pos_t_err"] - trimmed_data["neg_t_err"]) / 2 if "pos_t_err" in trimmed_data else (trimmed_data["timePos_sec"] - trimmed_data["timeNeg_sec"]) / 2
log_ts_error = ts_error / (ts * np.log(10))
err_time_params = st.norm.fit(log_ts_error)
recon_logtimeerr = st.norm(*err_time_params).rvs(len(log_recon_t))

df = trimmed_data.copy(deep=True)
for k in range(len(log_recon_t)):
    new_row = {
        "t": 10**log_recon_t[k][0],
        "pos_t_err": 10**recon_logtimeerr[k],
        "neg_t_err": 10**recon_logtimeerr[k],
        "flux": 10**log_reconstructed_flux[k],
        "pos_flux_err": 10**log_reconstructed_flux[k] * np.log(10) * recon_errorbar[k],
        "neg_flux_err": 10**log_reconstructed_flux[k] * np.log(10) * recon_errorbar[k]
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
df.to_csv(folder_path+f"{GRB_Name}_result.csv", index=False)

# 5 FOLD CROSS VALIDATION

kf = KFold(n_splits=5)

epoch = 0
epochs = 1000
patience = 50
patience_counter = 0
times = 5
best_mse = float("inf")
threshold = 0.1
pbar = tqdm(range(epochs))


kf_train_mse_losses = []
kf_test_mse_losses = []

while (epoch < epochs):
    epoch += 1

    for i, (train_index, test_index) in enumerate(kf.split(log_ts, log_fluxes)):

        train_ts, train_fluxes = log_ts[train_index], log_fluxes[train_index]
        test_ts, test_fluxes = log_ts[test_index], log_fluxes[test_index]

        train_ts=train_ts.reshape(-1,1)
        train_fluxes=train_fluxes.reshape(-1,1)

        model.fit(train_ts, train_fluxes, batch_size=batch_size, epochs=1, verbose=0)

        train_ts = train_ts.reshape(-1, 1)
        train_fluxes = train_fluxes.reshape(-1, 1)
        test_ts = test_ts.reshape(-1, 1)
        test_fluxes = test_fluxes.reshape(-1, 1)

        train_preds = model.predict(train_ts, batch_size=batch_size)
        train_mse_loss = np.mean((train_fluxes - train_preds) ** 2)

        test_preds = model.predict(test_ts, batch_size=batch_size)
        test_mse_loss = np.mean((test_fluxes - test_preds) ** 2)


        kf_train_mse_losses.append(train_mse_loss)
        kf_test_mse_losses.append(test_mse_loss)



    avg_test_loss = sum([kf_test_mse_losses[d] for d in range(-kf.get_n_splits(), 0, 1)])/kf.get_n_splits()
    pbar.update(1)
    pbar.set_description(f"Test Loss: {round(avg_test_loss, 4)}, Best Loss: {best_mse}, Patience: {patience_counter}")

    if round(avg_test_loss, 4) < best_mse:
        best_mse = round(avg_test_loss, 4)
        patience_counter = 0
    else:
        patience_counter += 1

# Reshape loss metrics for each fold
kf_train_mse_losses = np.array(kf_train_mse_losses).reshape(-1, kf.get_n_splits())
kf_test_mse_losses = np.array(kf_test_mse_losses).reshape(-1, kf.get_n_splits())


# file_path = f"/path/to/your/results/MSE/{GRB_Name}_MSE.csv"
file_path = f"{GRB_Name}_MSE.csv"
write_header = not os.path.exists(file_path)  # Only write the header if the file doesn't exist

data = {
    "GRB Name": [GRB_Name],
    "Final Train MSE": [kf_train_mse_losses],
    "Final Validation MSE":[kf_test_mse_losses]
}

df = pd.DataFrame(data)

df.to_csv(file_path, header=write_header, index=False)

# %%

