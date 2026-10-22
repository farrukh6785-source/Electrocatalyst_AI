"""Retrain the thesis ML models for ElectroCatalyst AI (v9).

Recipe (faithful port of the thesis FINAL models):
  - J_m0p7 (photocurrent at -0.7 V): StandardScaler + Lasso(alpha=0.05, max_iter=10000)
  - Rct_ohm, delta_Rct, tau_s: StandardScaler + GaussianProcessRegressor
      kernel = Constant * Matern(nu=1.5) + WhiteKernel, alpha=1e-10,
      normalize_y=True, n_restarts_optimizer=3

Validation (honest, leakage-free): LeaveOneGroupOut over the 28
sample x condition groups (3 voltage rows each). Row-wise scores are NOT
reported because sibling voltage rows leak.

Outputs: ml_bundle.pkl with models, feature lists, training medians /
ranges (for app defaults and out-of-range warnings), grouped LOOCV scores
and top drivers (LASSO coefficients for J, permutation importance for GPR).
"""
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "MASTER_TABLE_n84.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "ml_bundle.pkl"

BASE_FEATURES = [
    "metal_num", "loading_num", "condition_num", "voltage",
    "Rs_ohm", "Rct_ohm", "peak_freq_Hz", "tau_s", "phase_min", "Cdl_est_F",
    "Rct_dark", "delta_Rct", "Rct_ratio", "delta_Rs",
    "dDeltaRct_dV", "Rct_ratio_mean", "delta_Rct_mean",
    "Vfb_V", "Eg_eV", "E_onset_V", "|tafel|_mV_dec",
    "CA_dJ_mean_mA", "CA_dJ_std_mA", "CA_spike_mA", "CA_stability",
    "TEM_mean_nm", "TEM_std_nm", "XRD_gCN_tau_nm", "XRD_gCN_layers",
    "XRD_metal_tau_nm", "SSPL_quenching_ratio",
    "TRPL_tau_A_ns", "TRPL_tau_int_ns", "UV_Eg_eV",
]

TARGETS = {
    "J_m0p7": {"label": "Photocurrent J at -0.7 V (mA/cm2)", "model": "lasso"},
    "Rct_ohm": {"label": "Charge transfer resistance Rct (ohm)", "model": "gpr"},
    "delta_Rct": {"label": "LSPR resistance drop delta Rct (ohm)", "model": "gpr"},
    "tau_s": {"label": "Relaxation time tau (s)", "model": "gpr"},
}


def make_model(kind):
    if kind == "lasso":
        return make_pipeline(
            StandardScaler(), Lasso(alpha=0.05, max_iter=10000)
        )
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(1.0, (1e-2, 1e2), nu=1.5)
        + WhiteKernel(1e-2, (1e-6, 1e1))
    )
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42,
        ),
    )


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {CSV_PATH}: {df.shape}")
    groups_all = (df["sample"].astype(str) + "_" + df["condition"].astype(str)).values

    bundle = {
        "features_base": BASE_FEATURES,
        "targets": {},
        "encodings": {
            "metal": {"Ag": 0, "Au": 1, "C3N4": 2},
            "condition": {"Dark": 0, "410 nm": 1, "440 nm": 2, "Full spectrum": 3},
            "loading": {"ref": 0.0, "0.25": 0.25, "0.5": 0.5, "1.0": 1.0},
        },
        "medians": {},
        "ranges": {},
    }
    for f in BASE_FEATURES:
        bundle["medians"][f] = float(df[f].median())
        bundle["ranges"][f] = [float(df[f].min()), float(df[f].max())]

    logo = LeaveOneGroupOut()
    for target, info in TARGETS.items():
        feats = [f for f in BASE_FEATURES if f != target]
        tmp = df[feats + [target, "sample", "condition"]].dropna().reset_index(drop=True)
        g = (tmp["sample"].astype(str) + "_" + tmp["condition"].astype(str)).values
        X = tmp[feats].values
        y = tmp[target].values

        model = make_model(info["model"])
        model.fit(X, y)

        # Grouped LOOCV (honest score)
        y_pred = np.zeros(len(y))
        for tr, te in logo.split(X, y, groups=g):
            m = make_model(info["model"])
            m.fit(X[tr], y[tr])
            y_pred[te] = m.predict(X[te])
        r2 = r2_score(y, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))

        # Drivers
        drivers = []
        if info["model"] == "lasso":
            coefs = model.named_steps["lasso"].coef_
            order = np.argsort(-np.abs(coefs))
            for idx in order:
                if coefs[idx] != 0:
                    drivers.append({"feature": feats[idx], "value": float(coefs[idx])})
        else:
            perm = permutation_importance(
                model, X, y, n_repeats=5, random_state=42, scoring="r2"
            )
            order = np.argsort(-perm.importances_mean)
            for idx in order[:10]:
                drivers.append(
                    {
                        "feature": feats[idx],
                        "value": float(perm.importances_mean[idx]),
                    }
                )

        bundle["targets"][target] = {
            "label": info["label"],
            "model_type": info["model"],
            "features": feats,
            "model": model,
            "grouped_loocv": {
                "r2": float(r2),
                "rmse": rmse,
                "mae": mae,
                "n": int(len(y)),
                "n_groups": int(len(np.unique(g))),
            },
            "drivers": drivers,
        }
        print(
            f"{target}: model={info['model']} n={len(y)} groups={len(np.unique(g))} "
            f"grouped LOOCV R2={r2:.3f} RMSE={rmse:.4g} MAE={mae:.4g}"
        )

    joblib.dump(bundle, OUT_PATH)
    print(f"Saved bundle -> {OUT_PATH}")


if __name__ == "__main__":
    main()
