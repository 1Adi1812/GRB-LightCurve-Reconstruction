# IMPORTS
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from scipy.optimize import curve_fit
import scipy.stats as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kan import *

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse


parser = argparse.ArgumentParser(description="Process GRB Name as input.")
parser = argparse.ArgumentParser(description="Process GRB Name as input.")
parser.add_argument('--name', type=str, required=True, help="Specify the GRB Name")
parser.add_argument('--path', type=str, required=False, default="", help="Specify the path (optional)")
parser.add_argument('--norm', type=str, required=False, default="y")
parser.add_argument('--over', type=str, required=False, default="y")

args = parser.parse_args()

GRB_Name = str(args.name)

path = args.path
splits = 5

epochs = 350
batch_size = 64

normalize = [True if args.norm == "y" else False][0]
override = [True if args.over == "y" else False][0]


if not override and os.path.exists(f"Saved_Outputs/{GRB_Name}/{GRB_Name}.csv"):
    print("PREDICTION ALREADY EXISTS, EXITING...")
    exit()

os.makedirs(f"Saved_Outputs/{GRB_Name}", exist_ok=True)

print(f"\n{GRB_Name}\n")
# PREPROCESSING
print("PREPROCESSING...\n")
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

ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"]
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err

fluxes_err_sym= (pos_fluxes - neg_fluxes)/2

ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

# Defining gaps for reconstruction
gaps = np.diff(log_ts)
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


# DATASET
print("PREPARING FOR MODEL...\n")

log_ts = log_ts.reshape(-1, 1)
log_fluxes = log_fluxes.values.reshape(-1, 1)
log_recon_t = np.reshape(log_recon_t, (-1, 1))

if normalize:
    s_log_ts = MinMaxScaler((0, 1)).fit(log_ts)
    log_ts = s_log_ts.transform(log_ts)

    s_log_fx = MinMaxScaler((0, 1)).fit(log_fluxes)
    log_fluxes = s_log_fx.transform(log_fluxes)

    s_log_recon_t = s_log_ts
    log_recon_t = s_log_recon_t.transform(log_recon_t)

else:
    log_recon_t = log_recon_t


class Data(Dataset):
    def __init__(self, time: np.ndarray, flux: np.ndarray = None):
        self.time = torch.tensor(time, dtype=torch.float32)
        
        if flux is not None and len(flux) > 0:
            self.flag = True
            self.flux = torch.tensor(flux, dtype=torch.float32)
        else:
            self.flag = False
            self.flux = None 

    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, idx):
        if self.flag: 
            return self.time[idx], self.flux[idx]
        return self.time[idx]

# 5 FOLD CROSS VALIDATION

print("TRAINING...\n")
kf = KFold(n_splits=splits)

with open(f"Saved_Outputs/{GRB_Name}/logs.txt", "w") as file:
    file.write(f"{GRB_Name}\n")


train_mse, test_mse = [], []

lr = 1e-3
best_mse = float("inf")
best_model = None
criterion = nn.MSELoss()

for i, (train_index, test_index) in enumerate(kf.split(log_ts, log_fluxes)):
   model = KAN(width=[1, 5, 1], grid=5, k=9, seed=42).to("cuda")
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)

   train_loader = DataLoader(Data(log_ts[train_index], log_fluxes[train_index]), batch_size=batch_size, shuffle=False)
   test_loader = DataLoader(Data(log_ts[test_index], log_fluxes[test_index]), batch_size=batch_size, shuffle=False)

    pbar = tqdm(range(epochs))
    for epoch in range(epochs):
        train_preds, train_trues = [], []
        for time, flux in train_loader:
            time, flux = time.to("cuda", non_blocking=True), flux.to("cuda", non_blocking=True)

            predictions = model(time)
            loss = criterion(flux, predictions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_preds += predictions.detach().cpu().numpy().reshape(-1).tolist()
            train_trues += flux.detach().cpu().numpy().reshape(-1).tolist()

        train_preds = s_log_fx.inverse_transform(np.array(train_preds).reshape(-1, 1))
        train_trues = s_log_fx.inverse_transform(np.array(train_trues).reshape(-1, 1))

        train_mse.append(mean_squared_error(train_preds, train_trues))


        test_preds, test_trues = [], []
        model.eval()
        with torch.no_grad():
            for time, flux in test_loader:
                time, flux = time.to("cuda", non_blocking=True), flux.to("cuda", non_blocking=True)
        
                predictions = model(time)
                test_preds += predictions.detach().cpu().numpy().reshape(-1).tolist()
                test_trues += flux.detach().cpu().numpy().reshape(-1).tolist()
    
            test_preds = s_log_fx.inverse_transform(np.array(test_preds).reshape(-1, 1))
            test_trues = s_log_fx.inverse_transform(np.array(test_trues).reshape(-1, 1))

            test_mse.append(mean_squared_error(test_preds, test_trues))
            pbar.update(1)
            pbar.set_description(f"Train MSE: {train_mse[-1]:.5f}, Test MSE: {test_mse[-1]:.5f}")

np.save(f"Saved_Outputs/{GRB_Name}/train_mse.npy", np.array(train_mse))
np.save(f"Saved_Outputs/{GRB_Name}/test_mse.npy", np.array(test_mse))

#TRAINING MODEL
model = KAN(width=[1, 5, 1], grid=5, k=9, seed=42).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

pbar = tqdm(range(epochs))
for _ in range(epochs):
    train_loader = DataLoader(Data(log_ts, log_fluxes), batch_size=batch_size, shuffle=False)
    for time, flux in train_loader:
        time, flux = time.to("cuda", non_blocking=True), flux.to("cuda", non_blocking=True)

        predictions = model(time)
        loss = criterion(flux, predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(1)
        
        pbar.set_description(f"Train Loss: {loss.item():.4f}")

print(log_recon_t.shape)

recon_fluxes_up = []
test_loader = DataLoader(Data(log_recon_t))
with torch.no_grad():
    for time in test_loader:
        # print(time.shape)
        time = time.to("cuda", non_blocking=True)

        predictions = model(time)
        recon_fluxes_up += predictions.detach().cpu().numpy().reshape(-1).tolist()

log_ts = s_log_ts.inverse_transform(log_ts).squeeze()
log_fluxes = s_log_fx.inverse_transform(log_fluxes).squeeze()
log_recon_t = s_log_recon_t.inverse_transform(log_recon_t).squeeze()
recon_fluxes_up = s_log_fx.inverse_transform(np.array(recon_fluxes_up).reshape(-1, 1)).squeeze().tolist()

#ADDING NOISE

## JIGGLED POINTS
print("JIGGLING POINTS...\n")
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))


errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_errorbar=np.abs(err_dist.rvs(size=len(log_recon_t)))

#Point specific noise
point_specific_noise = []
for j in range(len(recon_fluxes_up)):
    fitted_dist = st.norm(loc=recon_fluxes_up[j], scale=recon_errorbar[j])
    point_noise = fitted_dist.rvs() - recon_fluxes_up[j]
    point_specific_noise.append(point_noise)

point_specific_noise = np.array(point_specific_noise)

#Jiggle reconstructed points
jiggled_points = recon_fluxes_up + point_specific_noise
jiggled_points = np.squeeze(jiggled_points)



## REALIZATIONS 
print("CALCULAING CONFIDENCE INTERVAL...\n")

num_samples = 1000  # Number of realizations
recon_fluxes_up = np.array(recon_fluxes_up)  
recon_errorbar = np.array(recon_errorbar)

point_specific_noise = np.random.normal(
    loc=0, scale=recon_errorbar, size=(num_samples, len(recon_fluxes_up))
)
jiggled_realizations = recon_fluxes_up + point_specific_noise
jiggled_realizations = jiggled_realizations

mean_jiggled = np.mean(jiggled_realizations, axis=0)
ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)  # 2.5th percentile
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)  # 97.5th percentile


## PLOTTING
print("SAVING PLOT...\n")

plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow', label = "Reconstructed Points")
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], linestyle="")
plt.scatter(log_ts, log_fluxes, label="Observed Points", zorder=5)
plt.plot(log_recon_t, recon_fluxes_up, label="Mean prediction", zorder=2)
plt.fill_between(
    log_recon_t, 
    np.squeeze(ci_95_lower), 
    np.squeeze(ci_95_upper), 
    alpha=0.5, 
    label='95% Confidence Interval')
plt.legend(loc='lower left')
plt.xlabel('log Time (s)', fontsize="15")
plt.ylabel("log Flux ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="15")
plt.title(f'KAN on {GRB_Name}', fontsize=18) 
plt.savefig(f'Saved_Outputs/{GRB_Name}/{GRB_Name}.png', dpi=300)


#CALCULATING TIME ERROR IN LINEAR SCALE
print("SAVING DATAFRAME...")
ts_error = (positive_ts_err - negative_ts_err)/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t)) # len(log_ts_error)
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


df.to_csv(f"Saved_Outputs/{GRB_Name}/{GRB_Name}.csv")

