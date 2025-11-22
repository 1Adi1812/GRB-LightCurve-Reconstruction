# Gamma-Ray Burst Light-curve Reconstruction: A Comparative Machine and Deep Learning Analysis

This repository contains all code used in the paper:

**Manchanda et al., "Gamma-Ray Burst Light-curve Reconstruction: A Comparative Machine and Deep Learning Analysis" (2025).**

The code implements a comparative study of multiple machine learning and deep learning methods for reconstructing gamma-ray burst (GRB) light curves.

---

## Abstract

Gamma-ray bursts (GRBs), observed at high redshift (high-z), are probes of the evolution of the Universe and can be used as cosmological tools. Correlations among key parameters require minimal dispersion. To reduce such dispersion, we mitigate gaps in light curves (LCs), including the plateau region, which is key to building the two-dimensional Dainotti relation between the end time of plateau emission (Tₐ) and its luminosity (Lₐ).

We reconstruct LCs using nine models:  

- Multilayer Perceptron (MLP)  
- Bi-Mamba  
- Fourier Transform  
- Gaussian Process–Random Forest Hybrid  
- Bidirectional Long Short-Term Memory (LSTM)  
- Conditional Generative Adversarial Networks (cGAN)  
- SARIMAX-based Kalman Filter  
- Kolmogorov–Arnold Networks (KAN)  
- Attention U-Net  

These methods are compared to the Willingale model (W07) over a sample of 521 GRBs.  

**Key results:**  

- **MLP** reduces plateau parameter uncertainties by 37.2%, 38.0%, and 41.2% for the plateau flux, plateau luminosity, and post-plateau slope α, achieving the lowest fivefold cross-validation mean squared error (MSE) of 0.0275.  
- **Attention U-Net** achieves the lowest parameter uncertainties (37.9%, 38.5%, 41.4%) but with a higher MSE of 0.134.  
- Other models yield MSE values ranging from 0.0339 to 0.174.  

Although Attention U-Net achieves the largest uncertainty reduction, MLP attains the lowest test MSE while maintaining comparable uncertainty performance, making it the more reliable model. These improvements are essential for using GRBs as standard candles, investigating theoretical models, and predicting GRB redshifts through machine learning.

---

## Repository Structure

All code for each method is contained in a **single Python script per model**, which includes:

- Data preprocessing  
- Model architecture  
- Training  
- Evaluation  
- Plotting   
## Installation

Follow these steps to set up your environment and install all dependencies:

### **Clone the repository**

```bash
git clone https://github.com/1Adi1812/GRB-LightCurve-Reconstruction
cd GRB-Lightcurve-Reconstruction

### **Create a virtual environment**

python3 -m venv env
source env/bin/activate   # Linux/macOS

### **Install required packages**

pip install --upgrade pip
pip install -r requirements.txt

## Contributors

This section lists the authors of each script in this repository.
Model                Author
GP                M. Dainotti
MLP                J. Felix
Bi-LSTM          A. Manchanda
Attention_UNet    N. Indoriya
GP_RF              S. Naqi
Bi-MAMBA          A. Kaushal
CGAN              A. Kaushal
Sarimax           A. Deepu
KAN              S.P. Magesh
Fourier           A. Deepu


## Citation

If you use this code, please cite:
Citation A. Manchanda et al 2025 ApJS 281 35

## Contact

For questions or suggestions, contact:
Aditi Manchanda – aditii18m@gmail.com



