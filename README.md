
**GlobalFitter-MKID** is a comprehensive, physics-driven global fitting framework designed for the automated and scalable characterization of frequency-division multiplexed (FDM) superconducting resonator arrays, such as Microwave Kinetic Inductance Detectors (MKIDs).

Traditional segmented fitting methods often fail or produce unphysical parameter scatter when dealing with severe baseline distortions, standing waves, or overlapping resonance tails. This algorithm overcomes these limitations by jointly modeling the entire wideband transmission spectrum ($S_{21}$), enforcing strict physical constraints, and utilizing advanced numerical optimization techniques.

## ✨ Key Features

* **Physics-Driven Hard Constraints:** Enforces a globally shared intrinsic quality factor ($Q_i$) across the entire array (reflecting uniform substrate TLS loss) and strictly accounts for Fano-resonance asymmetry via the $\cos(\phi)$ impedance projection.
* **Complex Baseline Calibration:** Utilizes an Asymmetric Least Squares (ASLS) smoothing strategy to dynamically extract and flatten severe nonlinear amplitude/phase baselines and electrical delays ($\tau$) without needing an explicit parametric model.
* **Extreme Numerical Stability:** Implements full parameter space normalization to prevent gradient divergence across variables with vastly different orders of magnitude.
* **Avoids "Baseline Dilution":** Introduces dynamic Gaussian weighting during optimization to heavily prioritize the critical resonance notches over the abundant off-resonance baseline points.
* **Highly Scalable & Fast:** Employs fully vectorized Jacobians and an *Adaptive Decimation* strategy (downsampling flat baselines while preserving notch bottoms) to achieve ultra-fast convergence even for arrays with hundreds of pixels.
* **Automated Spurious Peak Rejection:** Acts as a self-correcting diagnostic tool that automatically identifies and filters out false noise peaks based on physically implausible $Q_c$ outputs.

## 🚀 Why Global Fitting?
Compared to conventional independent circle fitting, this global approach:
1. Breaks the mathematical degeneracy between the global electrical delay, constant phase offset, and local asymmetry ($\phi$).
2. Eliminates non-physical $Q_i$ and $Q_c$ scatter induced by local measurement artifacts (e.g., cavity absorption peaks).
3. Requires zero manual tuning of threshold windows, enabling a fully automated characterization pipeline.

## 📖 Citation
If you use this code in your research, please consider citing our accompanying paper submitted to *Superconductor Science and Technology (SUST)*:
> *(Update this section with your paper's arXiv link or DOI once published)*
