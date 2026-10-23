# ElectroCatalyst AI

A local Streamlit app for electrochemical and photoelectrochemical
water-splitting analysis. Upload raw measurement files and get fitted
parameters, figures and interpretation in seconds. Everything runs on
your own machine: no data leaves your laptop.

## What it analyses

- **LSV (linear sweep voltammetry):** onset potential, overpotential at
  10 and 100 mA/cm2, Tafel slope with fit quality, maximum current
  density.
- **EIS (electrochemical impedance spectroscopy):** simple RC circuit
  fit (Rs, Rct, C) and CPE circuit fit (Rs, Rct, Q, n) with Nyquist,
  Bode and residual plots.
- **Chronoamperometry:** retention, average current density, duration.
- **Chronopotentiometry:** initial, final and mean potential, and
  potential drift in mV/h.
- **PEC (photoelectrochemical) interpretation:** dark vs light
  photocurrent, photocurrent at 1.23 V vs RHE, onset shift, ABPE,
  dark vs light EIS (PEIS), transient photocurrent and IPCE.
- **ML Prediction:** predicts current density, charge transfer
  resistance and time constant from fitted descriptors and catalyst
  settings.
- **Comparison dashboard:** rank analysed samples side by side, but
  only when electrolyte, pH, potential scale and light condition match.

## Install and run

Requirements: Python 3.12 or newer.

```bash
pip install -r requirements.txt
streamlit run electrocatalyst_real_app.py
```

On Windows, double-click `Run Real App.bat` instead. The app opens in
your browser at `http://localhost:8501`.

## How to use

1. Upload LSV, EIS, chronoamperometry or chronopotentiometry files
   (CS Studio `.txt` or `.xlsx` workbooks).
2. Set the current unit and the potential scale in the sidebar. If the
   scale is unknown, overpotential values are shown as "Not available"
   instead of guessed.
3. Read the results tables and the interpretation sections. Figures
   can be downloaded as PNG or TIF.

## ML Prediction: honest notes

The ML models are trained on 7 photocatalysts and 84 readings from an
FAU Erlangen-Nuremberg thesis dataset. Grouped leave-one-out validation
(28 sample and condition groups) gives R2 of 0.533 for current density
(moderate, treat as indicative), 0.930 for Rct, 0.929 for the change in
Rct and 0.906 for tau. Tafel slope is deliberately not predicted: the
thesis benchmark found no reliable signal for it.

The trained bundle `ml_bundle.pkl` is not stored in this repository
because it is a binary file. Regenerate it with:

```bash
python train_ml_models.py
```

This needs the thesis dataset file `MASTER_TABLE_n84.csv` next to the
script. Without the bundle, the ML Prediction section is unavailable,
but all other analysis works normally.

## Files in this repository

| File | Purpose |
| --- | --- |
| `electrocatalyst_real_app.py` | The Streamlit app (all analysis and presentation) |
| `unit_conversions.py` | Current and current density conversions |
| `test_units.py` | Tests for the conversions |
| `train_ml_models.py` | Retrains the ML models and rebuilds the bundle |
| `requirements.txt` | Python dependencies |
| `Run Real App.bat` | Windows launcher |
| `ipce_example.csv` | Example IPCE data to try the IPCE section |

## Notes

- Three-electrode photocurrent is not solar-to-hydrogen efficiency.
  Real efficiency needs a two-electrode device measurement with
  measured gas production.
- Results depend on the settings you choose, such as electrode area
  and potential scale. Change a setting and the headline numbers can
  move, so report the settings with the results.
