
# GlobalFitter-MKID (Ultimate Edition, Soft Shared Qi)

**GlobalFitter-MKID** is a physics-driven global fitting framework for automated characterization of superconducting resonator arrays, such as Microwave Kinetic Inductance Detectors (MKIDs) operating in frequency-division multiplexing (FDM) systems.

This *Ultimate Edition* introduces a **soft-shared intrinsic quality factor (Qi) constraint**, enabling robust fitting of realistic experimental data where perfect uniformity across resonators does not strictly hold.

---

## 🚀 Overview

Unlike traditional segmented or circle fitting methods, this framework jointly models the entire wideband transmission response:

S21(f) = A · exp[i(α + 2π f τ)] × Π_i [ 1 − (Ql_i / Qc_i) · exp(i φ_i) / (1 + 2i Ql_i (f − fr_i)/fr_i) ]

This approach ensures:

- Global physical consistency  
- Robustness against strong baseline distortions  
- Stability in the presence of overlapping resonances  
- Elimination of parameter degeneracy  

---

## ✨ Key Features

### ✅ Soft Shared Qi (Core Upgrade)

Instead of enforcing a strictly identical Qi for all resonators, we introduce:

log(Qi_i) = log(Qi_center) + Δ_i

with a **soft penalty** added to the optimizer:

penalty_i = w · (Δ_i / σ)

This provides:

- Controlled variation between resonators  
- Robust performance on non-ideal datasets  
- Suppression of unphysical Qi divergence  

**Key parameters:**
- `CFG_SOFT_QI_SIGMA_LOG` → allowed spread  
- `CFG_SOFT_QI_WEIGHT` → penalty strength  
- `CFG_SOFT_QI_MAX_DELTA_SIGMA` → hard cap  

---

### ✅ Physics-Based Parameter Model

Each resonator is described using:

- fr — resonance frequency  
- Ql — loaded quality factor  
- Qc — coupling quality factor  
- Qi — intrinsic quality factor  
- φ — Fano asymmetry phase  

With the exact physical relationship:

1/Qi = 1/Ql − cos(φ)/Qc

---

### ✅ Dynamic Baseline Calibration (ASLS)

The framework automatically removes:

- Electrical delay (τ)  
- Standing waves  
- Nonlinear amplitude distortion  
- Phase drift  

This is achieved using **Asymmetric Least Squares (ASLS)** without requiring any explicit baseline model.

---

### ✅ Global Reparameterization & Normalization

To improve numerical stability, the optimization is performed on:

**Global variables:**
[A, alpha, tau_residual, logQi_center]

**Per-resonator variables:**
[fr, Qc, phi, dlogQi]

**Derived internally:**
- Qi = exp(logQi)
- Ql⁻¹ = Qi⁻¹ + cos(phi)/Qc

Benefits:
- Prevents gradient explosion  
- Improves convergence  
- Removes parameter degeneracy  

---

### ✅ Adaptive Downsampling (High Performance)

Strategy:

- Keep **all data points inside resonance windows**
- Downsample flat baseline regions

Result:

- ~80–95% reduction in data size  
- No loss in fitting accuracy  
- Significantly faster optimization  

---

### ✅ Smart Weighting (Anti-Baseline Dilution)

Baseline regions dominate least-squares fitting numerically.

Solution:

weights = max(0.01, Gaussian resonance weights)

Effects:

- Resonances dominate optimization  
- Baseline still constrains global structure  

---

### ✅ Automatic Resonance Detection

- Peak detection based on:
  - prominence  
  - height threshold  
  - minimum Q constraint  

- Adaptive window sizing (based on linewidth)

- Overlapping windows are automatically flagged:
  `[OVERLAP]`

---

### ✅ Strict Physical Constraints

Each parameter is tightly bounded:

- fr: limited within linewidth-based margin  
- Qc: bounded relative to initial estimate  
- φ: restricted within small range  
- Qi: controlled globally (soft + bounds)  

Ensures:

- Physical validity  
- Numerical stability  
- Prevention of divergence  

---

## 🔄 Fitting Pipeline

### Step 1 – Preprocessing
- Estimate electrical delay (τ)  
- Remove baseline using ASLS  

### Step 2 – Resonance Detection
- Identify peaks  
- Build resonance windows  

### Step 3 – Independent Initialization
- Fit each resonance locally  
- Extract initial fr, Ql, φ, k  

### Step 4 – Adaptive Downsampling
- Reduce redundant data points  

### Step 5 – Global Optimization

Using:

scipy.optimize.least_squares

Features:

- Parameter normalization  
- Complex residual (real + imaginary parts)  
- Soft Qi penalty  
- Weighted residual  

---

## 📊 Output Parameters

Each resonator returns:

- fr (Hz) — resonance frequency  
- Ql — loaded quality factor  
- Qi — intrinsic quality factor  
- Qc — coupling quality factor  
- phi — asymmetry phase  
- k = Ql/Qc — coupling ratio  
- dlogQi — deviation from global Qi  

Global outputs:

- Qi_center_soft  
- baseline parameters (A, alpha, tau)  
- residual error  

---

## 🔥 Advantages Over Traditional Methods

### Compared to Circle Fitting

- No phase-delay degeneracy  
- No manual window selection  
- Works with overlapping resonances  

### Compared to Independent Fitting

- Eliminates Qi scatter  
- Enforces global physics  
- Uses full dataset information  

---

## ⚡ Performance

- Supports **10–200+ resonators**
- Handles:
  - Severe baseline distortion  
  - Strong resonance overlap  
  - Low SNR  

Typical runtime:

- ~10–100 seconds (with downsampling)

---

## 📖 Citation

If you use this code in your research, please cite:

(doi:10.1088/1361-6668/ae9f9b)

---

## ✅ Summary

**GlobalFitter-MKID (Ultimate Edition)** is a robust, physically constrained, and numerically stable global fitting framework featuring a **soft-shared Qi model**, designed to reliably extract resonator parameters from real-world MKID measurements where traditional methods fail.
