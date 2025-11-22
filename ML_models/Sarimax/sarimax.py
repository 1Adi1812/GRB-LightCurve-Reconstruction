#IMPORTS
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy import stats as st
from sklearn.model_selection import KFold
from tqdm import tqdm

base_dir = "/path/to/your/data/"
input_dir = os.path.join(base_dir, "GRBs_trimmed")
csv_out_dir = os.path.join(base_dir, "sarimax/csv")
img_out_dir = os.path.join(base_dir, "sarimax/images")
train_mse_out_dir = os.path.join(base_dir, "sarimax/train_mse")
test_mse_out_dir = os.path.join(base_dir, "sarimax/test_mse")

# Function to run the model for a given GRB
def run_model(GRB_NAME):
    # Read the CSV file for the specific GRB
    input_path = os.path.join(input_dir, f"{GRB_NAME}_trimmed.csv")
    output_csv_path = os.path.join(csv_out_dir, f"{GRB_NAME}_sarimax.csv")
    output_img_path = os.path.join(img_out_dir, f"{GRB_NAME}_sarimax.png")
    trimmed_data = pd.read_csv(input_path)

    trim_t_val = trimmed_data["t"] if "t" in trimmed_data else trimmed_data["time_sec"]
    ts, fluxes = trim_t_val.to_numpy(), trimmed_data["flux"].to_numpy()
    flux2 = fluxes
    ts2 = ts
    ts, fluxes = np.log10(ts), np.log10(fluxes)
    gaps = np.diff(ts)

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

    upsampled_times = []
    upsampled_fluxes = []
    for i in range(len(ts) - 1):
        start_time = ts[i]
        end_time = ts[i + 1]
        gap = end_time - start_time
        if gap > 0.2:
            new_times = np.linspace(start_time, end_time, num=7)[1:-1]
            upsampled_times.extend(new_times)
    upsampled_times = np.array(upsampled_times)
    interpolation_function = interp1d(ts, fluxes, kind='linear', fill_value="extrapolate")
    fluxes_new = interpolation_function(upsampled_times)

    # Skip first plot (Actual Light Curves)
    plt.figure()
    plt.plot(ts, fluxes, 'b.', alpha=0.5)
    plt.scatter(upsampled_times, fluxes_new)
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title(f'Actual Light Curves - {GRB_Name}')
    plt.show()
    plt.close()

    all_times = np.concatenate([ts, upsampled_times])
    all_fluxes = np.concatenate([fluxes, fluxes_new])
    sorted_indices = np.argsort(all_times)
    all_times = all_times[sorted_indices]
    all_fluxes = all_fluxes[sorted_indices]

    arima_order = (1, 1, 1)
    model = SARIMAX(all_fluxes, order=arima_order)
    fitted_model = model.fit(disp=False)
    kalman_results = fitted_model.smoother_results
    fluxes_imputed = np.nan * np.ones_like(recon_t)
    smoothed_fluxes = kalman_results.smoothed_state[0]
    interpolation_function = interp1d(all_times, smoothed_fluxes, kind='linear', fill_value="extrapolate")
    fluxes_imputed = interpolation_function(log_recon_t)
    interpolation_function2 = interp1d(log_recon_t, fluxes_imputed, kind='linear', fill_value="extrapolate")
    fluxes_imputed_for_missing = interpolation_function2(ts)
    fluxes_imputed_for_missing = np.copy(fluxes)
    fluxes_imputed_for_missing[np.isnan(fluxes)] = fluxes_imputed_for_missing[np.isnan(fluxes)]

    positive_fluxes_err = trimmed_data["flux_errPos"] if "flux_errPos" in trimmed_data else trimmed_data["pos_flux_err"]
    negative_fluxes_err = trimmed_data["flux_errNeg"] if "flux_errNeg" in trimmed_data else trimmed_data["neg_flux_err"]
    pos_fluxes = flux2 + positive_fluxes_err
    neg_fluxes = flux2 + negative_fluxes_err
    pos_log_fluxes = np.log10(pos_fluxes)
    neg_log_fluxes = np.log10(neg_fluxes)
    fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2
    logfluxerrs = (fluxes_error) / (flux2 * np.log(10))
    errparameters = st.norm.fit(logfluxerrs)
    err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])
    recon_errorbar = err_dist.rvs(size=len(log_recon_t))

    point_specific_noise = []
    for j in range(len(fluxes_imputed)):
        fitted_dist = norm(loc=fluxes_imputed[j], scale=recon_errorbar[j])
        point_noise = fitted_dist.rvs() - fluxes_imputed[j]
        point_specific_noise.append(point_noise)
    point_specific_noise = np.array(point_specific_noise)
    new = fluxes_imputed + point_specific_noise

    from scipy.ndimage import gaussian_filter1d
    num_samples = 500
    jiggled_realizations = []
    for _ in range(num_samples):
        point_specific_noise = []
        for j in range(len(fluxes_imputed)):
            fitted_dist = norm(loc=fluxes_imputed[j], scale=recon_errorbar[j])
            point_noise = fitted_dist.rvs() - fluxes_imputed[j]
            point_specific_noise.append(point_noise)
        jiggled_realizations.append(fluxes_imputed + np.array(point_specific_noise))
    jiggled_realizations = np.array(jiggled_realizations)
    mean_jiggled = np.mean(jiggled_realizations, axis=0)
    ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)
    ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)
    smooth_ci_95_lower = gaussian_filter1d(ci_95_lower, sigma=2)
    smooth_ci_95_upper = gaussian_filter1d(ci_95_upper, sigma=2)
    smooth_log_recon_t = log_recon_t[:len(smooth_ci_95_lower)]

    pos_log_fluxes = np.log10(pos_fluxes)
    neg_log_fluxes = np.log10(neg_fluxes)

    # Second plot: SARIMAX Results
    plt.figure()
    plt.errorbar(log_recon_t, new, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow', label = "Reconstructed Points",zorder=3)
    plt.errorbar(ts, fluxes, yerr=[fluxes-neg_log_fluxes,pos_log_fluxes-fluxes], linestyle="",zorder=4)
    # plt.errorbar(ts, fluxes, yerr=[fluxes-neg_log_fluxes, pos_log_fluxes-fluxes], linestyle="")
    plt.scatter(ts, fluxes, label="Observed Points",zorder=5)
    plt.plot(log_recon_t, mean_jiggled, label="Mean prediction",zorder=2)
    plt.fill_between(
        smooth_log_recon_t.ravel(),
        smooth_ci_95_lower,
        smooth_ci_95_upper,
        alpha=0.5,
        label=r"95% confidence interval",
        zorder=1
    )
    plt.legend(loc='lower left')
    plt.xlabel('log$_{10}$(Time) (s)', fontsize="15")
    plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)", fontsize="15")
    plt.title(f"SARIMAX on {GRB_Name}", fontsize=18)
    plt.savefig(output_img_path, dpi=300)
    plt.show()
    plt.close()

    mse_fn = interp1d(log_recon_t, new, kind='linear', fill_value="extrapolate")
    flux3 = mse_fn(ts)
    print(f"Mean Squared Error for {GRB_Name}: {mean_squared_error(flux3, fluxes)}")

    positive_ts_err = trimmed_data["timePos_sec"] if "timePos_sec" in trimmed_data else trimmed_data["pos_t_err"]
    negative_ts_err = trimmed_data["timeNeg_sec"] if "timeNeg_sec" in trimmed_data else trimmed_data["neg_t_err"]
    ts_error = (positive_ts_err - negative_ts_err) / 2
    log_ts_error = ts_error / (ts2 * np.log(10))
    errparameters = st.norm.fit(log_ts_error)
    err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])
    recon_logtimeerr = err_dist_time.rvs(size=len(log_recon_t))

    imputed_data = pd.DataFrame({
        "t": 10**log_recon_t,
        "pos_t_err": 10**recon_logtimeerr,
        "neg_t_err": 10**recon_logtimeerr,
        "flux": 10**new,
        "pos_flux_err": 10**new * np.log(10) * recon_errorbar,
        "neg_flux_err": 10**new * np.log(10) * recon_errorbar,
    })

    # Save the imputed data to a CSV file in the specified folder
    imputed_data.to_csv(output_csv_path, index=False)

    # 5 FOLD CROSS VALIDATION

    def fit_predict_arima(model, train_times, train_fluxes, test_times):
        results = model.fit(disp=False)  # Fit the model
        kalman_results = results.smoother_results
        smoothed_fluxes = kalman_results.smoothed_state[0]

        interpolation_function = interp1d(train_times, smoothed_fluxes, kind='linear', fill_value="extrapolate")
        predictions = interpolation_function(test_times)

        return results, predictions

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
    model = SARIMAX(all_fluxes, order=(1, 1, 1))

    kf_train_mse_losses = []
    kf_test_mse_losses= []

    while (epoch < epochs):
        epoch += 1

        for i, (train_index, test_index) in enumerate(kf.split(all_times, all_fluxes)):

            train_ts, train_fluxes = all_times[train_index], all_fluxes[train_index]
            test_ts, test_fluxes = all_times[test_index], all_fluxes[test_index]

            results, train_preds = fit_predict_arima(model, all_times, all_fluxes, train_ts)
            train_mse_loss = np.mean((train_fluxes - train_preds) ** 2)

            _, test_preds = fit_predict_arima(model, log_ts, log_fluxes, test_ts)
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
    df3 = pd.DataFrame(kf_train_z_losses)
    df4 = pd.DataFrame(kf_test_z_losses)

    df1.to_csv(f'{train_mse_out_dir}/{GRB_NAME}_train_mse.csv')
    df2.to_csv(f'{test_mse_out_dir}/{GRB_NAME}_test_mse.csv')
    
    print("Cross Validation done")

GRB_Name = 'GRB050820A'

run_model(GRB_Name)

