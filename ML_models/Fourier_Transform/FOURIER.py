import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy import stats as st
from sklearn.model_selection import KFold
from tqdm import tqdm

# Read in the data
header_names = ['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
GRB_Name = 'GRB050803'
trimmed_data = pd.read_csv(f'/path/to/your/GRB/data/{GRB_Name}_trimmed.csv', 
                           skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)

# Extract time and flux data
ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"].to_numpy()

# Convert to log scale
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

# min gap 0.05, assign points based on the number of original points
min_gap = 0.05
recon_log_t = [ts[0]]
total_span = ts[-1] - ts[0]
if len(ts) > 500:  # densest LC
    fraction = 0.05
elif len(ts) > 250:
    fraction = 0.1
elif len(ts) > 100:
    fraction = 0.3
else:
    fraction = 0.4
n_points = max(20, int(fraction * len(ts)))
for i in range(len(ts) - 1):
    gap_size = ts[i+1] - ts[i]
    if gap_size > min_gap:
        interval_points = max(2, int(n_points * gap_size / total_span))
        interval = np.linspace(ts[i], ts[i+1], interval_points, endpoint=True)
        recon_log_t.extend(interval[1:])
recon_log_t = np.array(recon_log_t)
recon_t = 10**np.array(recon_log_t)
recon_t = np.unique(recon_t)
log_recon_t = np.log10(recon_t)

interp_fn = interp1d(log_ts, log_fluxes, kind='cubic', fill_value='extrapolate')
uniform_flux = interp_fn(log_recon_t)


fft_coeffs = np.fft.fft(uniform_flux)
freqs = np.fft.fftfreq(len(log_recon_t), d=np.diff(log_recon_t).mean())


cutoff = int(0.1 * len(fft_coeffs))
fft_coeffs[cutoff:-cutoff] = 0


fourier_recon = np.fft.ifft(fft_coeffs).real


fourier_interp = interp1d(log_recon_t, fourier_recon, kind='cubic', fill_value='extrapolate')
reconstructed_log_fluxes = fourier_interp(log_recon_t)

positive_fluxes_err = trimmed_data["pos_flux_err"]
negative_fluxes_err = trimmed_data["neg_flux_err"]


pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err
pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

fluxes_error=(positive_fluxes_err-negative_fluxes_err)/2

logfluxerrs=(fluxes_error)/(fluxes*np.log(10))

errparameters=st.norm.fit(logfluxerrs)
err_dist=st.norm(loc=errparameters[0], scale=errparameters[1])
recon_errorbar=err_dist.rvs(size=len(log_recon_t))
point_specific_noise=[]
            # print(recon_errorbar)
for j in range(len(reconstructed_log_fluxes)):


    fitted_dist=norm(loc=reconstructed_log_fluxes[j], scale=recon_errorbar[j])
    point_noise=fitted_dist.rvs() - reconstructed_log_fluxes[j]
    point_specific_noise.append(point_noise)

point_specific_noise = np.array(point_specific_noise)
new = reconstructed_log_fluxes + point_specific_noise

from scipy.ndimage import gaussian_filter1d

num_samples = 1000
jiggled_realizations = []

for _ in range(num_samples):
    point_specific_noise = []
    for j in range(len(reconstructed_log_fluxes)):
        fitted_dist = norm(loc=reconstructed_log_fluxes[j], scale=recon_errorbar[j])
        point_noise = fitted_dist.rvs() - reconstructed_log_fluxes[j]
        point_specific_noise.append(point_noise)
    jiggled_realizations.append(reconstructed_log_fluxes + np.array(point_specific_noise))

jiggled_realizations = np.array(jiggled_realizations)

            # Compute mean and 95% confidence intervals
mean_jiggled = np.mean(jiggled_realizations, axis=0)
ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)  # 2.5th percentile
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0) 

            # Apply Gaussian filter to the confidence interval bounds
smooth_ci_95_lower = gaussian_filter1d(ci_95_lower, sigma=2)  # Adjust sigma for smoothing
smooth_ci_95_upper = gaussian_filter1d(ci_95_upper, sigma=2)

smooth_log_recon_t = log_recon_t[:len(smooth_ci_95_lower)]

fn=interp1d(log_recon_t,reconstructed_log_fluxes, kind='linear', fill_value="interpolate")
fl=fn(log_ts)
print(mean_squared_error(fl,reconstructed_log_fluxes))

positive_ts_err = trimmed_data["pos_t_err"]
negative_ts_err = trimmed_data["neg_t_err"]
ts_error = (positive_ts_err - negative_ts_err)/2

    #CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))


errparameters = st.norm.fit(log_ts_error) 
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t))
print(ts_error)

imputed_data = pd.DataFrame({
                "t": 10**log_recon_t,  
                    "pos_t_err": 10**recon_logtimeerr,
                "neg_t_err": 10**recon_logtimeerr,
                "flux": 10**new,
                "pos_flux_err": 10**new* np.log(10) * recon_errorbar,
                "neg_flux_err": 10**new * np.log(10) * recon_errorbar,
                
            })

# Save the imputed data to a CSV file in the specified folder
imputed_data.to_csv(f"/path/to/your/results/{GRB_Name}.csv" index=False)

plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], label=r"$\log_{10}\,flux$", linestyle="", zorder=4)
plt.errorbar(log_recon_t, new, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow', zorder=3)

plt.scatter(log_ts, log_fluxes, label='Observations', zorder=5)
plt.plot(log_recon_t, new, label='Mean predictions', zorder=2)
plt.fill_between(
    smooth_log_recon_t.ravel(),
    smooth_ci_95_lower,
    smooth_ci_95_upper,
    alpha=0.5,
    label=r"95% confidence interval",
    zorder=1
)

plt.xlabel('log Time', fontsize=18) 
plt.ylabel('log Flux', fontsize=18) 
plt.title(f'Fourier on {GRB_Name}', fontsize=18) 

plt.tick_params(axis='both', labelsize=14)

plt.legend(fontsize=14)
plt.savefig(f"/path/to/your/results/{GRB_Name}.png", dpi=300, bbox_inches='tight')

# 5 FOLD CROSS VALIDATION
def fit_predict_fourier(train_times, train_fluxes, test_times):
    interp_fn = interp1d(train_times, train_fluxes, kind='cubic', fill_value='extrapolate')
    uniform_flux = interp_fn(test_times)
    fft_coeffs = np.fft.fft(uniform_flux)

    cutoff = int(0.1 * len(fft_coeffs))
    if cutoff > 0:
        fft_coeffs[cutoff:-cutoff] = 0

    filtered_flux = np.fft.ifft(fft_coeffs).real
    log_test_ts = np.log10(test_times)
    fourier_interp = interp1d(test_times, filtered_flux, kind='cubic', fill_value='extrapolate')
    predictions = fourier_interp(test_times)

    return predictions
    

kf = KFold(n_splits=5)

# @tf.function()
epoch = 0
epochs = 1
patience = 50
patience_counter = 0
times = 5
best_mse = float("inf")
threshold = 0.1
pbar = tqdm(range(epochs))

# Define model

kf_train_mse_losses = []
kf_test_mse_losses= []

while (epoch < epochs):
    epoch += 1

    for i, (train_index, test_index) in enumerate(kf.split(log_ts, log_fluxes)):

        train_ts, train_fluxes = log_ts[train_index], log_fluxes[train_index]
        test_ts, test_fluxes = log_ts[test_index], log_fluxes[test_index]

        train_preds = fit_predict_fourier(log_ts, log_fluxes, train_ts)
        train_mse_loss = np.mean((train_fluxes - train_preds) ** 2)

        test_preds = fit_predict_fourier(log_ts, log_fluxes, test_ts)
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


    if patience_counter >= patience:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_test_loss:.4f}")

        if avg_test_loss > threshold and times != 0:
            print(f"Thresholding at epoch {epoch+1} due to no improvement in validation loss")
            epochs += patience
            times -= 1
            patience_counter = 0

        else:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
            break

kf_train_mse_losses = np.array(kf_train_mse_losses).reshape(-1, kf.get_n_splits())
kf_test_mse_losses = np.array(kf_test_mse_losses).reshape(-1, kf.get_n_splits())


df1 = pd.DataFrame(kf_train_mse_losses)
df2 = pd.DataFrame(kf_test_mse_losses)

df1.to_csv(f'/path/to/your/results/{GRB_NAME}_train_mse.csv')
df2.to_csv(f'/path/to/your/results/{GRB_NAME}_test_mse.csv')

print("Cross Validation done")

