# GlobalFitter-MKID

**GlobalFitter-MKID** is a Python framework for automated characterization of superconducting microwave resonator arrays, including Microwave Kinetic Inductance Detectors (MKIDs).

The program analyzes wideband complex transmission data and fits multiple resonators together in a single global model. It is designed to make resonator-array analysis more robust, reproducible, and efficient than manually fitting each resonance independently.

The physical model, fitting strategy, and validation are described in the accompanying paper:

> Please cite: [doi:10.1088/1361-6668/ae9f9b](https://doi.org/10.1088/1361-6668/ae9f9b)

## Highlights

- **Automated multi-resonator fitting** from wideband complex S21 data.
- **Global analysis** that uses information from the complete measurement band.
- **Robust baseline handling** for data affected by electrical delay, standing waves, and phase or amplitude distortions.
- **Automatic resonance detection** and adaptive fitting-window construction.
- **Support for overlapping resonances** and challenging low-SNR measurements.
- **Physically meaningful parameter extraction**, including resonance frequency and quality factors.
- **Soft global Qi constraint** that reduces unphysical scatter while allowing realistic resonator-to-resonator variation.
- **Adaptive data reduction** to improve runtime without discarding important resonance information.
- **Batch-friendly command-line tools** with optional plotting and CSV export.

For the detailed equations, parameterization, optimization strategy, and performance evaluation, please refer to the paper rather than the implementation summary here.

## What the program does

The typical analysis workflow is:

1. Read experimental or simulated complex S21 data.
2. Estimate and correct the measurement baseline.
3. Detect resonances automatically.
4. Initialize individual resonator parameters.
5. Fit the resonator array globally.
6. Save fitted parameters and diagnostic figures.

The main per-resonator outputs are:

- Resonance frequency (`fr`)
- Loaded quality factor (`Ql`)
- Internal quality factor (`Qi`)
- Coupling quality factor (`Qc`)
- Asymmetry phase (`phi`)
- Coupling ratio (`k`)
- Deviation from the global Qi estimate (`dlogQi`)

## Installation

From the repository root, install the runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

Python 3.9 or newer is recommended.

## Quick start

### Experimental data

Use `main_test_experiment_data_v7.py` for Touchstone or text measurement data:

```bash
python main_test_experiment_data_v7.py path/to/measurement.s2p
python main_test_experiment_data_v7.py path/to/measurement.s2p --no-plot
```

The experimental-data script writes a timestamped CSV file next to the input file:

```text
fit_results_YYYYMMDD_HHMMSS.csv
```

When plotting is enabled, it also saves `global_robust_fit.png` in the current working directory.

### Simulation data

Use `main_test_simulation_data_v7.py` for simulation data:

```bash
python main_test_simulation_data_v7.py S21_simulation.txt
python main_test_simulation_data_v7.py S21_simulation.txt --no-plot
```

To generate an example simulation input file:

```bash
python generate_simulation_data.py
```

The simulation fitting script saves its default figure as `global.png` next to the input file.

## Input data

Touchstone `.s2p` files are read with `scikit-rf`. The fitter uses the complex `S21` parameter.

Plain-text input files must contain at least two whitespace-separated columns:

1. Frequency in Hz
2. Complex S21

Comment lines may begin with `#`. For example:

```text
# Frequency(Hz) S21_complex
5.000000000000e+09 9.500000000000e-01+1.200000000000e-02j
5.000100000000e+09 9.490000000000e-01+1.250000000000e-02j
```

## Repository structure

- `main_test_experiment_data_v7.py` — entry point for experimental data.
- `main_test_simulation_data_v7.py` — entry point for simulation data.
- `generate_simulation_data.py` — example simulation-data generator.
- `main_data_reader.py` — readers for Touchstone and text S21 data.
- `main_single_resonator_v1.py` — local single-resonator fitting utilities.
- `main_circle_fit.py`, `main_phase_fit.py` — supporting fitting modules.

## Notes

The default analysis settings are intended to provide a useful starting point for typical datasets. Depending on the measurement quality and resonator spacing, resonance-detection thresholds, minimum-Q settings, and fitting-window parameters may need adjustment.

Plotting is enabled by default for interactive analysis. Use `--no-plot` for headless or batch execution.

## License

This project is released under the [MIT License](LICENSE).
