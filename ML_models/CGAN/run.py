#IMPORTS
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import scipy.stats as st
from tqdm import tqdm
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description="Process GRB Name as input.")
parser.add_argument('--name', type=str, required=True, help="Specify the GRB Name")
parser.add_argument('--path', type=str, required=False)
parser.add_argument('--ep', type=int, required=False)
args = parser.parse_args()

GRB_Name = args.name

path = args.path
if path is None:
    path = ""

epochs = args.ep
if epochs is None:
    EPOCHS = 25000
else:
    EPOCHS = epochs


os.makedirs(f"Saved_Outputs/{GRB_Name}", exist_ok=True)


## PREPROCESSING
header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']

trimmed_data = pd.read_csv(f"/path/to/your/GRB/data/{GRB_Name}_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)
trimmed_data=trimmed_data.sort_values(by="t").reset_index(drop=True)


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

ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"]

#ABOVE VALUES IN LOG SCALE
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

#ERROR ON THE FLUXES
pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err

#SYMMETRIC ERROR VALUE
fluxes_err_sym= ( pos_fluxes - neg_fluxes )/2

#CALCULATING ERRORBAR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

#Defining gaps for reconstruction
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

log_recon_t = np.log10(recon_t).reshape(-1)


### CGAN MODEL

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Generator (Conv1D version)
# -------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim):
        super(Generator, self).__init__()
        input_dim = noise_dim + cond_dim
        self.dense1 = nn.Linear(input_dim, 256)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1d_1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv1d(64, 16, kernel_size=3, padding=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(16 * 256, 1)

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=-1)
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = x.view(x.size(0), 1, -1)
        x = self.relu_conv1(self.conv1d_1(x))
        x = self.relu_conv2(self.conv1d_2(x))
        x = self.flatten(x)
        return self.output_layer(x)


# -----------------------------
# Discriminator (Conv1D version)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, data_dim, cond_dim):
        super(Discriminator, self).__init__()
        input_dim = data_dim + cond_dim
        self.dense1 = nn.Linear(input_dim, 256)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1d_1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv1d(64, 16, kernel_size=3, padding=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(16 * 256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, condition):
        x = torch.cat([data, condition], dim=-1)
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = x.view(x.size(0), 1, -1)
        x = self.relu_conv1(self.conv1d_1(x))
        x = self.relu_conv2(self.conv1d_2(x))
        x = self.flatten(x)
        return self.sigmoid(self.output_layer(x))


# -------------------------
# Training helpers & loop
# -------------------------
def train_step(
    generator, discriminator,
    d_optimizer, g_optimizer,
    real_data, condition,
    criterion, NOISE_DIM
):
    """
    one batch step. real_data and condition are torch tensors on device
    criterion: BCELoss (expects probabilities since discriminator ends with Sigmoid)
    returns: d_loss.item(), g_loss.item()
    """
    batch_size = real_data.size(0)

    # CREATE NOISE uniform [0,1)
    noise = torch.rand(batch_size, NOISE_DIM, device=device)

    # ---------------------
    # Train Discriminator
    # ---------------------
    discriminator.train()
    generator.eval()  # freeze generator while training D (behaviorally similar to your TF code within tape contexts)

    # real labels - ones, fake labels - zeros
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # real forward
    real_output = discriminator(real_data, condition)   # (batch,1)
    # generate fake
    with torch.no_grad():
        fake_data = generator(noise, condition)
    fake_output = discriminator(fake_data, condition)

    d_loss_real = criterion(real_output, real_labels)
    d_loss_fake = criterion(fake_output, fake_labels)
    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # ---------------------
    # Train Generator
    # ---------------------
    generator.train()
    discriminator.eval()

    # We want generator to fool discriminator => target ones
    # Reuse same noise (like TF code uses same noise in g pass)
    noise = torch.rand(batch_size, NOISE_DIM, device=device)  # TF re-generated noise once per batch in original; you may reuse same noise if desired
    g_optimizer.zero_grad()
    fake_data = generator(noise, condition)
    fake_output = discriminator(fake_data, condition)
    g_loss = criterion(fake_output, real_labels)  # targets are ones

    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item()



log_fluxes = log_fluxes.values
s_flux = MinMaxScaler((0, 1))
s_ts = MinMaxScaler((0, 1))

s_log_fluxes = s_flux.fit_transform(np.expand_dims(log_fluxes, axis=1))
s_log_ts = s_ts.fit_transform(np.expand_dims(log_ts, axis=1))
s_log_recon_ts = s_ts.fit_transform(np.expand_dims(log_recon_t, axis=1))

# Hyperparameters
BATCH_SIZE = 256
NOISE_DIM = 10
LEARNING_RATE = 1e-4



def generate_data(X_np, y_np, gen, s_ts=None, NOISE_DIM=10):
    """
    X_np: numpy array for condition (time) shape (n, cond_dim)
    y_np: unused here (kept for API parity)
    gen: generator model (in eval mode)
    s_ts: scaler for inverse_transform (sklearn scaler)
    returns: (time_inverse_transformed, flux_numpy_squeezed)
    """
    gen.eval()
    n = X_np.shape[0]
    noise = torch.rand(n, NOISE_DIM, device=device)
    condition = torch.tensor(X_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        generated = gen(noise, condition).cpu().numpy()  # shape (n,1) or (n,)
    flux = np.squeeze(generated)  # match TF: np.squeeze

    time = None
    if s_ts is not None:
        # note: in your TF you passed `condition.numpy()` then inverse transformed with s_ts
        time = s_ts.inverse_transform(X_np)
    else:
        time = X_np

    return time, flux


def train(X, y, EPOCHS=500, BATCH_SIZE=256, NOISE_DIM=10, LEARNING_RATE=1e-4):
    """
    X: numpy array conditions (time)
    y: numpy array real_data (flux)
    """
    # Convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # dataset expects (real_data, condition) similar to TF Dataset.from_tensor_slices((log_fluxes_T, log_ts_T))
    dataset = TensorDataset(y_t, X_t)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # initialize models
    cond_dim = X.shape[1] if X.ndim > 1 else 1
    data_dim = y.shape[1] if y.ndim > 1 else 1
    generator = Generator(NOISE_DIM, cond_dim).to(device)
    discriminator = Discriminator(data_dim, cond_dim).to(device)

    # After creating models, we need to ensure their dense layers have correct in_features by doing a dummy forward
    # This is needed because we lazily init dense layers based on input dim.
    # Make a small dummy forward to trigger initialization:
    dummy_noise = torch.rand(1, NOISE_DIM, device=device)
    dummy_condition = torch.rand(1, X_t.shape[1] if X_t.ndim > 1 else 1, device=device)
    # ensure shape of condition is (1, cond_dim)
    if dummy_condition.ndim == 1:
        dummy_condition = dummy_condition.unsqueeze(0)
    # forward passes to initialize Dense layers
    _ = generator(dummy_noise, dummy_condition)
    _ = discriminator(y_t[:1].to(device), dummy_condition)

    # optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # loss - Keras used BinaryCrossentropy with discriminator output probabilities because discriminator has sigmoid
    criterion = nn.BCELoss()

    print("TRAINING...\n")
    for epoch in range(EPOCHS):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batches = 0
        for real_data_batch, condition_batch in dataloader:
            real_data_batch = real_data_batch.to(device)
            condition_batch = condition_batch.to(device)

            d_loss, g_loss = train_step(
                generator=generator,
                discriminator=discriminator,
                d_optimizer=d_optimizer,
                g_optimizer=g_optimizer,
                real_data=real_data_batch,
                condition=condition_batch,
                criterion=criterion,
                NOISE_DIM=NOISE_DIM
            )
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            batches += 1

        if epoch % 50 == 0:
            avg_d = epoch_d_loss / max(1, batches)
            avg_g = epoch_g_loss / max(1, batches)
            print(f"Epoch {epoch}, Discriminator Loss: {avg_d:.4f}, Generator Loss: {avg_g:.4f}")

    return generator, discriminator


# 5 FOLD CROSS VALIDATION
kf = KFold(n_splits=5)

with open(f"Saved_Outputs/{GRB_Name}/logs.txt", "w") as file:
    file.write(f"{GRB_Name}\n")

train_mse, test_mse = [], []
for i, (train_index, test_index) in enumerate(kf.split(log_ts, log_fluxes)):
    print(f"CV Split: {i}")
    X_train, y_train = s_log_ts[train_index, ...], log_fluxes[train_index][..., np.newaxis]
    X_test, y_test = s_log_ts[test_index, ...], log_fluxes[test_index][..., np.newaxis]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    generator = train(X_train, y_train)
    
    _, train_gen_fluxes = generate_data(X_train, y_train, gen=generator)
    _, test_gen_fluxes = generate_data(X_test, y_test, gen=generator)
    
      # plt.scatter(X_test.squeeze(-1), test_gen_fluxes, c="r", alpha=0.5)
#     # plt.scatter(X_test.squeeze(-1), y_test.squeeze(-1), c="b", alpha=0.5)
#     # plt.savefig(f"Saved_Outputs/{GRB_Name}/{GRB_Name}.png", dpi=300, bbox_inches='tight')

    with open(f"Saved_Outputs/{GRB_Name}/logs.txt", "a") as file:
        file.write(f"CV Split: {i}: \n")
        tr_mse = mean_squared_error(train_gen_fluxes, y_train.squeeze(-1))
        te_mse = mean_squared_error(test_gen_fluxes, y_test.squeeze(-1))
        file.write(f"Train MSE: {tr_mse:.4f}\n")
        file.write(f"Test MSE: {te_mse:.4f}\n")

        train_mse.append(tr_mse)
        test_mse.append(te_mse)


np.save(f"Saved_Outputs/{GRB_Name}/train_mse.npy", np.array(train_mse))
np.save(f"Saved_Outputs/{GRB_Name}/test_mse.npy", np.array(test_mse))
        

print(log_ts.shape, log_fluxes.shape, log_recon_t.shape)
generator, discriminator = train(
    log_ts[..., np.newaxis], 
    log_fluxes[..., np.newaxis], 
    EPOCHS=EPOCHS, BATCH_SIZE=128, NOISE_DIM=10, LEARNING_RATE=1e-4)
time, recon_fluxes_up = generate_data(log_recon_t[..., np.newaxis], log_fluxes, generator)

print(log_fluxes.shape, recon_fluxes_up.shape)

## JIGGLED POINTS
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
plt.title(f'CGAN on {GRB_Name}', fontsize=18) 
plt.savefig(f'Saved_Outputs/{GRB_Name}/{GRB_Name}.png', dpi=300)
plt.show()


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
