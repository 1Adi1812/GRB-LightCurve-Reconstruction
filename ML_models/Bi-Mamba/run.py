# IMPORTS
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from scipy.optimize import curve_fit
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
from utils import HybridLoss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mambular.base_models.utils.basemodel import BaseModel
from model import MambaRegressor

GRB_Name = "GRB070508"


lr = 1e-4
patience = 30
batch_size = 64

epochs = 100
threshold = 0.01
times = 5

os.makedirs(f"/path/to/your/results/{GRB_Name}", exist_ok=True)

# PREPROCESSING
print("PREPROCESSING...\n")
header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
trimmed_data = pd.read_csv(f"/path/to/your/data/{GRB_Name}_trimmed.csv", skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)
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


# Dataset
print("PREPARING FOR MODEL...\n")
class Data(Dataset):
    def __init__(self, time: np.ndarray, flux: np.ndarray):
        self.time = torch.tensor(time, dtype=torch.float32).view(-1, 1)
        self.flux = torch.tensor(flux, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, idx):
        return self.time[idx], self.flux[idx]
    
class SingleData(Dataset):
    def __init__(self, time: np.ndarray):
        self.A = torch.tensor(time, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.A.shape[0]
    
    def __getitem__(self, idx):
        return self.A[idx]
    

log_ts = log_ts.reshape(-1, 1)
log_fluxes = log_fluxes.values.reshape(-1, 1)
log_recon_t = np.reshape(log_recon_t, (-1, 1))

if normalize:
    s_log_ts = MinMaxScaler((0, 1)).fit(log_ts)
    log_ts = s_log_ts.transform(log_ts)

    s_log_fx = MinMaxScaler((0, 1)).fit(log_fluxes)
    log_fluxes = s_log_fx.transform(log_fluxes)

    log_recon_t = s_log_ts.transform(log_recon_t)

else:
    log_recon_t = log_recon_t


### TODO: REMOVE THIS IF NOT NEEDED
# indices = np.random.choice(log_recon_t.shape[0], size=int(0.4*log_recon_t.shape[0]), replace=False)
# print(log_recon_t.shape[0])
# log_recon_t = log_recon_t[indices]
# print(len(indices))


# gaps = np.diff(log_ts)


# Model training

model = MambaRegressor(
    output_dim=1,
    dt_rank=32,
    d_state=48,
    d_model=128,
    expand_factor=80,
    dropout=0.1,
    bidirectional=True,
    activation=nn.LeakyReLU()
).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loader = DataLoader(Data(log_ts, log_fluxes), batch_size=batch_size, shuffle=False)

pbar = tqdm(range(epochs))
for epoch in range(epochs):
    train_preds, train_trues = [], []
    for time, flux in train_loader:
        time, flux = time.reshape(-1, 1, 1).to("cuda", non_blocking=True), flux.reshape(-1, 1, 1).to("cuda", non_blocking=True)

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

    pbar.update(1)
    pbar.set_description(f"Train MSE: {train_mse[-1]:.4f}")

 
# EVALUATING
print("EVALUATING...\n")
recon = SingleData(log_recon_t)
recon = DataLoader(recon, batch_size=batch_size, shuffle=False)

model.eval()
log_recon_fx = []
with torch.no_grad(): 
    for recon_time in recon:
        recon_time = torch.reshape(recon_time, (-1, 1, 1)).to("cuda")

        val_predictions = model(recon_time)
        log_recon_fx.extend(val_predictions.detach().cpu().numpy().reshape(-1))

log_recon_fx = np.reshape(log_recon_fx, (-1, 1))

if normalize:
    log_ts = s_log_ts.inverse_transform(log_ts)
    log_fluxes = s_log_fx.inverse_transform(log_fluxes)

    log_recon_t = s_log_ts.inverse_transform(log_recon_t)
    log_recon_fx = s_log_fx.inverse_transform(log_recon_fx)

log_fluxes = np.squeeze(log_fluxes)
log_recon_t = np.squeeze(log_recon_t)
recon_fluxes_up = np.squeeze(log_recon_fx)

## Adding Noise
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
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], linestyle="", zorder=4)
plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow',label = "Reconstructed Points", zorder=3)

plt.scatter(log_ts, log_fluxes, label='Observed Points', zorder=5)
plt.plot(log_recon_t, mean_prediction, label='Mean predictions', zorder=2)
plt.fill_between(log_recon_t, np.squeeze(ci_95_lower), np.squeeze(ci_95_upper), color='orange', alpha=0.5, label='95% Confidence Interval', zorder=1)

plt.legend(loc='lower left', fontsize = "10")
plt.xlabel('log$_{10}$(Time) (s)',fontsize="15")
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="15")
plt.title(f'BiMAMBA on {GRB_Name}', fontsize=18)

plt.tick_params(axis='both', labelsize=14)

plt.savefig(f"Saved_Outputs/{GRB_Name}/{GRB_Name}.png", dpi=300, bbox_inches='tight')


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


# 5 FOLD CROSS VALIDATION

kf = KFold(n_splits=5)

with open(f"Saved_Outputs/{GRB_Name}/logs.txt", "w") as file:
    file.write(f"{GRB_Name}\n")

best_loss = float('inf')
patience_counter = 0  
epochs = 120
train_mse, test_mse = [], []

best_mse = float("inf")
best_model = None
criterion = nn.MSELoss()

for i, (train_index, test_index) in enumerate(kf.split(log_ts, log_fluxes)):
     model = MambaRegressor(
         output_dim=1,
         dt_rank=32,
         d_state=48,
         d_model=128,
         expand_factor=80,
         dropout=0.1,
         bidirectional=True,
         activation=nn.LeakyReLU()
     ).to("cuda")
     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

     train_loader = DataLoader(Data(log_ts[train_index], log_fluxes[train_index]), batch_size=batch_size, shuffle=False)
     test_loader = DataLoader(Data(log_ts[test_index], log_fluxes[test_index]), batch_size=batch_size, shuffle=False)

     pbar = tqdm(range(epochs))
     for epoch in range(epochs):
         train_preds, train_trues = [], []
         for time, flux in train_loader:
             time, flux = time.reshape(-1, 1, 1).to("cuda", non_blocking=True), flux.reshape(-1, 1, 1).to("cuda", non_blocking=True)

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
                 time, flux = time.reshape(-1, 1, 1).to("cuda", non_blocking=True), flux.reshape(-1, 1, 1).to("cuda", non_blocking=True)

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
