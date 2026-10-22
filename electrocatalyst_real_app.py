# -*- coding: utf-8 -*-
"""ElectroCatalyst AI, REAL single file web app (Streamlit), version 3.

Stage 1 corrections:
  1. Chronoamperometry time is hours from seconds: seconds / 3600.
  2. Current density uses unit_conversions.to_current_density. A
     density column is only converted between A and mA, never divided
     by area again. Area is used only when the current itself is in
     A or mA. The current unit can be chosen in the sidebar.
  3. Potential reference scale is explicit: Unknown, RHE, or Other
     with a verified offset. When the scale is Unknown, overpotential
     is not calculated and a warning is shown.
  4. Tafel fit interval can be chosen in the sidebar. The fit reports
     slope, R2, points used and the fitted interval, with a warning
     when the fit is weak.
  5. EIS uses a simple RC circuit fit (Rs + (Rct || C)) with RMSE.
     When the fit fails, no parameters are reported. Nothing is
     substituted, and fitted values are model dependent.

Stage 2 ML prediction (ml_bundle.pkl, trained by train_ml_models.py):
  Lasso for photocurrent J, GPR for Rct, delta Rct and tau, trained on
  7 photocatalysts x 84 readings. Scores shown are grouped
  leave-one-group-out scores, never in-sample fits.

Stage 3 comparison dashboard:
  Analysed samples can be added with their experimental conditions
  (electrolyte, pH, potential scale, light condition). Two entries count
  as comparable only when all conditions match. Exact duplicates are
  refused, so uploading one file twice cannot move a ranking. Strict
  ranking leaves entries with missing conditions, and entries that are
  alone under their conditions, unranked.

Stage 4 photoelectrochemical (PEC) interpretation:
  From a dark LSV and a light LSV of the same sample: photocurrent is
  the light current minus the dark current at the same potential, plus
  onset shift and, when a measured incident power is entered, ABPE for
  anodic photocurrent. Photocurrent from a three electrode measurement
  is never presented as solar to hydrogen efficiency.

Stage 5 full PEC and fitting upgrades:
  Tafel Fit Inspector with points, fitted line and residuals; CPE
  circuit fit with Nyquist overlay, Bode plots and residuals plus arc
  closure and depression warnings; chronopotentiometry analysis;
  transient photocurrent from light on intervals; IPCE from measured
  wavelength dependent power; PEIS comparison of dark and light Rct.

v13 presentation UI (same analysis as v12, nothing else changed):
  Hero banner, component styling and headline metric cards only.
  Every analysis function, parser, fit and ML path is identical to v12.

Run with:  python -m streamlit run electrocatalyst_real_app.py
Then open the address it prints (usually http://localhost:8501).
"""

import io
import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except ImportError:
    least_squares = None
    _HAVE_SCIPY = False

from unit_conversions import to_current_density

# ============================================================================
# FILE READING
# ============================================================================

def _cs_id_from_text(text):
    """CS Studio experiment id from a header line, or None."""
    m = re.search(r"CSStudioFile,(ID_[A-Za-z0-9]+)", text)
    return m.group(1) if m else None


def read_text_table(file_bytes):
    """Read a .txt or .csv upload. Returns (DataFrame or None, cs_id)."""
    text = file_bytes.decode("utf-8", errors="ignore") if isinstance(file_bytes, (bytes, bytearray)) else file_bytes
    text = text.lstrip("\ufeff")
    cs_id = _cs_id_from_text(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, cs_id
    cleaned = [ln.strip().strip('"') for ln in lines]
    tabbed = [ln for ln in cleaned if "\t" in ln and "," not in ln]
    if len(tabbed) >= 10:
        rows = [ln.split("\t") for ln in tabbed]
        try:
            [float(x) for x in rows[0]]
            header = [f"col{i}" for i in range(len(rows[0]))]
            body = rows
        except ValueError:
            header, body = rows[0], rows[1:]
        ncol = max(len(r) for r in body) if body else 0
        cols = [header[i] if i < len(header) else f"col{i}" for i in range(ncol)]
        df = pd.DataFrame([r + [None] * (ncol - len(r)) for r in body], columns=cols)
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
        return df, cs_id
    if "\t" in lines[0].split(",")[0]:
        rows = []
        for ln in lines:
            parts = ln.split(",")[0].split("\t")
            if len(parts) >= 2:
                rows.append(parts)
        header, body = rows[0], rows[1:]
        ncol = max(len(r) for r in body) if body else 0
        cols = [header[i] if i < len(header) else f"col{i}" for i in range(ncol)]
        df = pd.DataFrame([r + [None] * (ncol - len(r)) for r in body], columns=cols)
    else:
        df = pd.read_csv(io.StringIO(text))
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    return df, cs_id


def read_excel_tables(file_bytes):
    """Read an .xlsx upload. Returns a list of (sheet name, DataFrame, cs_id)."""
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        return []
    out = []
    for sheet in xl.sheet_names:
        try:
            raw = xl.parse(sheet, header=None)
        except Exception:
            continue
        cs_id = None
        for val in raw.iloc[:, 0].astype(str).head(3):
            found = _cs_id_from_text(val)
            if found:
                cs_id = found
                break
        header_row = None
        for i, row in raw.iterrows():
            cells = [str(v).strip().lower() for v in row.tolist()]
            if any("e(v)" in c for c in cells) or any("freq" in c for c in cells):
                header_row = i
                break
        if header_row is None:
            continue
        header = [str(v).strip() for v in raw.iloc[header_row].tolist()]
        body = raw.iloc[header_row + 1:]
        body.columns = header
        df = body.reset_index(drop=True)
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
        out.append((sheet, df, cs_id))
    return out


def read_upload(uploaded):
    """Read one uploaded file. Returns a list of (sample name, DataFrame, cs_id)."""
    name = uploaded.name
    data = uploaded.read()
    if name.lower().endswith(".xlsx"):
        return [(f"{name} [{sheet}]", df, cs_id)
                for sheet, df, cs_id in read_excel_tables(data)]
    df, cs_id = read_text_table(data)
    return [(name, df, cs_id)] if df is not None else []


def pick_column(df, keywords, fallback_idx):
    cols = list(df.columns)
    low = [str(c).lower() for c in cols]
    for kw in keywords:
        for i, nm in enumerate(low):
            if kw in nm:
                return cols[i]
    return cols[min(fallback_idx, len(cols) - 1)]


def numeric_series(df, col):
    series = df[col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def resolve_current_unit(jcol, selected):
    """Map a current column and the sidebar selection to one of the four
    units understood by to_current_density. A density column is only
    converted between A and mA, never divided by area again. Area is
    used only when the current itself is in A or mA."""
    if selected != "auto":
        return selected
    name = str(jcol).lower()
    if "ma/cm" in name:
        return "mA/cm²"
    if "a/cm" in name:
        return "A/cm²"
    if "(ma)" in name or name.strip().endswith("ma"):
        return "mA"
    if "(a)" in name:
        return "A"
    return "mA/cm²"


def _colnames(df):
    return [str(c).lower() for c in df.columns]


def _has(df, *keywords):
    low = _colnames(df)
    return any(any(kw in nm for kw in keywords) for nm in low)


def _is_cyclic(potential):
    """True when the potential sweep reverses direction substantially, as in
    cyclic voltammetry. LSV sweeps one way only."""
    v = pd.Series(np.asarray(potential, dtype=float)).dropna().to_numpy()
    if len(v) < 50:
        return False
    w = max(5, len(v) // 100)
    if w % 2 == 0:
        w += 1
    sm = pd.Series(v).rolling(w, center=True, min_periods=1).median().to_numpy()
    d = np.diff(sm)
    signs = np.sign(d[np.abs(d) > 1e-9])
    if len(signs) < 10:
        return False
    span = float(np.max(sm) - np.min(sm))
    if span <= 0:
        return False
    reversals = 0
    last = signs[0]
    run_start = sm[0]
    for i in range(1, len(signs)):
        if signs[i] != last:
            excursion = abs(sm[i] - run_start)
            if excursion > 0.25 * span:
                reversals += 1
                run_start = sm[i]
            last = signs[i]
    return reversals >= 2


def detect_technique(df, cs_id):
    """Detect the technique from the data itself. Returns one of:
    LSV, EIS, Chrono, CP, CV, Unknown. LSV mode (HER or OER) is decided later
    from the current sign. CP is constant current with a varying
    potential, which is the reverse of chronoamperometry."""
    if df is None or len(df) < 10 or len(df.columns) < 2:
        return "Unknown"
    if _has(df, "z'", "z''", "zreal", "zimag", "impedance") or (
            _has(df, "freq") and _has(df, "z'", "real")):
        return "EIS"
    if cs_id == "ID_LSVA":
        return "LSV"
    if cs_id == "ID_PotStatic":
        return "Chrono"
    if cs_id == "ID_CV":
        return "CV"
    if not (_has(df, "e(v", "potential", "volt") and _has(df, "i(", "current", "dens")):
        return "Unknown"
    try:
        vcol = pick_column(df, ["e (v", "e(v", "potential", "volt"], 0)
        tcol = None
        for c in df.columns:
            if str(c).strip().lower() in ("t(s)", "t (s)", "time", "time (s)"):
                tcol = c
                break
        v = numeric_series(df, vcol).dropna().to_numpy(dtype=float)
        if len(v) < 10:
            return "Unknown"
        if tcol is not None:
            t = numeric_series(df, tcol).dropna().to_numpy(dtype=float)
            if len(t) >= 10 and float(np.max(v) - np.min(v)) < 0.02 and t[-1] > t[0]:
                return "Chrono"
            try:
                jcol = pick_column(df, ["i (", "i(", "current", "dens", "j (", "j("], 1)
                cur = numeric_series(df, jcol).dropna().to_numpy(dtype=float)
                mean_cur = float(np.mean(cur)) if len(cur) else 0.0
                cv_cur = (float(np.std(cur)) / abs(mean_cur)) if mean_cur != 0 else 999.0
                if (len(t) >= 10 and len(cur) >= 10 and t[-1] > t[0]
                        and float(np.max(v) - np.min(v)) > 0.1 and cv_cur < 0.05):
                    return "CP"
            except Exception:
                pass
        if _is_cyclic(v):
            return "CV"
        return "LSV"
    except Exception:
        return "Unknown"

# ============================================================================
# LSV
# ============================================================================

def parse_lsv(df, current_unit="auto", area_cm2=0.5, rhe_offset=None):
    """Returns (data, scale). Potential scale: a file RHE column wins,
    because it is already RHE. Otherwise rhe_offset None means the
    scale is unknown and overpotentials are not calculated; rhe_offset
    0.0 means the potentials are already RHE; any other value shifts
    recorded potentials by that offset (RHE = E + offset). Current
    density uses to_current_density: a density column is only
    converted between A and mA, never divided by area again."""
    rhe_col = None
    for c in df.columns:
        if str(c).strip().lower() == "rhe":
            rhe_col = c
            break
    if rhe_col is not None:
        v = numeric_series(df, rhe_col)
        scale = "RHE (from file)"
    else:
        vcol = pick_column(df, ["e (v", "e(v", "potential", "volt", "v vs"], 0)
        v = numeric_series(df, vcol)
        if "mv" in str(vcol).lower():
            v = v / 1000.0
        if rhe_offset is None:
            scale = "Unknown (not RHE)"
        elif rhe_offset == 0.0:
            scale = "RHE"
        else:
            v = v + rhe_offset
            scale = f"RHE (offset {rhe_offset:.4f} V)"
    jcol = pick_column(df, ["i (", "i(", "current", "dens", "j (", "j("], 1)
    j = numeric_series(df, jcol)
    unit = resolve_current_unit(jcol, current_unit)
    j = to_current_density(j.to_numpy(dtype=float), unit, area_cm2)
    data = pd.DataFrame({"Potential_V": v, "Current_Density": j}).dropna().reset_index(drop=True)
    return data, scale


def _up_crossings(v, j_abs, target):
    out = []
    for i in range(1, len(v)):
        if j_abs[i - 1] < target <= j_abs[i]:
            x0, x1 = j_abs[i - 1], j_abs[i]
            frac = 0.0 if x1 == x0 else (target - x0) / (x1 - x0)
            out.append(float(v[i - 1] + frac * (v[i] - v[i - 1])))
    if len(v) and j_abs[0] >= target:
        out.insert(0, float(v[0]))
    return out


def tafel_slope(v, j, j_min=None, j_max=None):
    """Fit overpotential against log10|j|. Returns (slope mV/dec, R2,
    points used, j interval low, j interval high). When j_min and
    j_max are given and valid, the fit uses that |j| interval;
    otherwise the 15 to 75 percent band of log10|j| is used. A fit
    with too few points returns slope None so nothing is claimed."""
    v = np.asarray(v, dtype=float)
    aj = np.abs(np.asarray(j, dtype=float))
    mask = (aj > 0) & np.isfinite(v) & np.isfinite(aj)
    v, aj = v[mask], aj[mask]
    if len(v) < 10:
        return None, None, 0, None, None
    logj = np.log10(aj)
    if j_min is not None and j_max is not None and 0 < j_min < j_max:
        sel = (aj >= j_min) & (aj <= j_max)
        lo_j, hi_j = float(j_min), float(j_max)
    else:
        lo, hi = np.quantile(logj, 0.15), np.quantile(logj, 0.75)
        sel = (logj >= lo) & (logj <= hi)
        lo_j, hi_j = float(10 ** lo), float(10 ** hi)
    if sel.sum() < 5:
        return None, None, int(sel.sum()), lo_j, hi_j
    slope, intercept = np.polyfit(v[sel], logj[sel], 1)
    if abs(slope) < 1e-9:
        return None, None, int(sel.sum()), lo_j, hi_j
    pred = slope * v[sel] + intercept
    ss_res = float(np.sum((logj[sel] - pred) ** 2))
    ss_tot = float(np.sum((logj[sel] - np.mean(logj[sel])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return float(1000.0 / abs(slope)), r2, int(sel.sum()), lo_j, hi_j


def analyze_lsv(data, mode="auto", rhe_ok=True, tafel_j_min=None, tafel_j_max=None):
    v = data["Potential_V"].to_numpy(dtype=float)
    j = data["Current_Density"].to_numpy(dtype=float)
    if mode == "auto":
        mode = "HER" if np.nanmax(-j) > np.nanmax(j) else "OER"
    order = np.argsort(v) if mode == "OER" else np.argsort(-v)
    v, j = v[order], j[order]
    aj = np.abs(j)
    c1 = _up_crossings(v, aj, 1.0)
    c10 = _up_crossings(v, aj, 10.0)
    c100 = _up_crossings(v, aj, 100.0)
    onset = c1[0] if c1 else None
    e10 = c10[-1] if c10 else None
    e100 = c100[-1] if c100 else None
    slope, r2, npts, jlo, jhi = tafel_slope(v, j, tafel_j_min, tafel_j_max)
    m = {
        "mode": mode,
        "onset_potential_V": onset,
        "E_at_10_mA_V": e10,
        "E_at_100_mA_V": e100,
        "overpotential_10_mV": None,
        "overpotential_100_mV": None,
        "tafel_slope_mV_dec": slope,
        "tafel_r2": r2,
        "tafel_points": npts,
        "tafel_j_min": jlo,
        "tafel_j_max": jhi,
        "max_abs_current_mA_cm2": float(np.nanmax(aj)) if len(aj) else None,
        "points": int(len(v)),
    }
    if rhe_ok:
        if mode == "HER":
            if e10 is not None:
                m["overpotential_10_mV"] = abs(e10) * 1000.0
            if e100 is not None:
                m["overpotential_100_mV"] = abs(e100) * 1000.0
        else:
            if e10 is not None:
                m["overpotential_10_mV"] = (e10 - 1.23) * 1000.0
            if e100 is not None:
                m["overpotential_100_mV"] = (e100 - 1.23) * 1000.0
    return m

# ============================================================================
# EIS
# ============================================================================

def parse_eis(df):
    zr_col = pick_column(df, ["z'", "zreal", "z real", "real", "zr"], 0)
    zi_col = pick_column(df, ["z''", "zimag", "z imag", 'z"', "imag", "zi"], 1)
    f_col = None
    for kw in ["freq", "hz"]:
        for c in df.columns:
            if kw in str(c).lower():
                f_col = c
                break
        if f_col:
            break
    out = pd.DataFrame({
        "Zreal": numeric_series(df, zr_col),
        "Zimag": numeric_series(df, zi_col),
        "Freq": numeric_series(df, f_col) if f_col else np.nan,
    }).dropna(subset=["Zreal", "Zimag"])
    return out.reset_index(drop=True)


def _rc_impedance(freq, rs, rct, cap):
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    return rs + rct / (1 + 1j * w * rct * cap)


def analyze_eis(data):
    """Simple RC circuit fit: Rs + (Rct || C). Frequency is used
    directly in the complex impedance fit. When the fit fails, or
    SciPy is not available, no parameters are reported. Nothing is
    substituted, and no fitted value is automatically called a
    physically proven charge transfer resistance."""
    zr = data["Zreal"].to_numpy(dtype=float)
    zi = data["Zimag"].to_numpy(dtype=float)
    freq = data["Freq"].to_numpy(dtype=float) if "Freq" in data else np.array([])
    result = {"Rs_ohm": None, "Rct_ohm": None, "C_F": None,
              "RMSE_ohm": None, "fit_x": None, "fit_y": None,
              "points": int(len(zr)), "fit_ok": False}
    ok = (len(zr) >= 6 and len(freq) == len(zr) and np.isfinite(freq).all()
          and np.all(freq > 0) and _HAVE_SCIPY)
    if not ok:
        return result
    z = zr + 1j * zi
    rs0 = float(np.min(zr))
    rct0 = max(float(np.max(zr) - np.min(zr)), 1.0)
    idx = int(np.argmax(-zi))
    w_peak = 2 * np.pi * freq[idx]
    c0 = 1.0 / (w_peak * rct0) if w_peak > 0 and rct0 > 0 else 1e-3

    def resid(p):
        rs, rct, cap = p
        fit = _rc_impedance(freq, rs, rct, cap)
        return np.concatenate([(fit.real - zr), (fit.imag - zi)])

    try:
        sol = least_squares(resid, [rs0, rct0, c0],
                            bounds=([0, 0, 1e-12], [np.inf, np.inf, np.inf]))
        rs, rct, cap = [float(x) for x in sol.x]
        if not all(np.isfinite([rs, rct, cap])) or rct <= 0 or cap <= 0:
            return result
        fit = _rc_impedance(freq, rs, rct, cap)
        rmse = float(np.sqrt(np.mean(np.abs(fit - z) ** 2)))
        result.update({"Rs_ohm": rs, "Rct_ohm": rct, "C_F": cap,
                       "RMSE_ohm": rmse, "fit_ok": True,
                       "fit_x": fit.real, "fit_y": -fit.imag})
    except Exception:
        return result
    return result

# ============================================================================
# CHRONOAMPEROMETRY
# ============================================================================

def parse_chrono(df, current_unit="auto", area_cm2=0.5):
    """Time is returned in hours: seconds divided by 3600. A T(s)
    column is treated as seconds. Current density uses
    to_current_density: a density column is only converted between
    A and mA, never divided by area again; area is used only when
    the current itself is in A or mA."""
    tcol = None
    convert_time = True
    for c in df.columns:
        if str(c).strip().lower() in ("t(s)", "t (s)"):
            tcol = c
            break
    if tcol is None:
        for c in df.columns:
            if str(c).strip().lower() in ("time", "time (s)"):
                tcol = c
                convert_time = False
                break
    if tcol is None:
        tcol = pick_column(df, ["time", "duration", "t ("], 0)
        convert_time = False
    jcol = pick_column(df, ["i (", "i(", "current", "dens", "j (", "j("], 1)
    j = numeric_series(df, jcol)
    unit = resolve_current_unit(jcol, current_unit)
    j = to_current_density(j.to_numpy(dtype=float), unit, area_cm2)
    t = numeric_series(df, tcol)
    if convert_time:
        t = t / 3600.0
    out = pd.DataFrame({
        "Time": t,
        "Current_Density": j,
    }).dropna()
    return out.sort_values("Time").reset_index(drop=True)


def analyze_chrono(data):
    t = data["Time"].to_numpy(dtype=float)
    j = np.abs(data["Current_Density"].to_numpy(dtype=float))
    n = len(t)
    k = max(1, n // 20)
    j_start = float(np.mean(j[:k]))
    j_end = float(np.mean(j[-k:]))
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    retention = (j_end / j_start * 100.0) if j_start > 0 else None
    return {
        "initial_current": j_start,
        "final_current": j_end,
        "retention_pct": retention,
        "avg_current": float(np.mean(j)) if n else None,
        "duration": duration,
        "points": int(n),
    }


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "Not available"
    return f"{x:.{digits}f}"


def safe_filename(name):
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return base.strip("._") or "figure"


def figure_bytes(fig, fmt, dpi):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def lsv_figure(data, m, curve_color, marker_color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["Potential_V"], data["Current_Density"], color=curve_color)
    if m["E_at_10_mA_V"] is not None:
        target = 10 if m["mode"] == "OER" else -10
        ax.plot([m["E_at_10_mA_V"]], [target], "o", color=marker_color)
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current density (mA/cm2)")
    ax.grid(True, alpha=0.3)
    return fig


def eis_figure(data, m, points_color, fit_color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["Zreal"], -data["Zimag"], "o", markersize=3, label="Measured", color=points_color)
    if m["fit_x"] is not None:
        ax.plot(m["fit_x"], m["fit_y"], label="RC model fit", color=fit_color)
    ax.set_xlabel("Z' (ohm)")
    ax.set_ylabel("minus Z'' (ohm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def chrono_figure(data, curve_color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["Time"], data["Current_Density"], color=curve_color)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Current density (mA/cm2)")
    ax.grid(True, alpha=0.3)
    return fig

# ============================================================================
# STREAMLIT UI (English only, for international presentation)
# ============================================================================

def interpret_lsv(results):
    """Plain language interpretation of LSV results. Works for one sample
    and for a comparison across several samples."""
    if not results:
        return []
    lines = []
    multi = len(results) > 1
    scored = {k: v for k, v in results.items()
              if v.get("overpotential_10_mV") is not None and v["overpotential_10_mV"] >= 0}
    if scored:
        best = min(scored, key=lambda k: scored[k]["overpotential_10_mV"])
        mode = scored[best]["mode"]
        if multi:
            lines.append(
                f"Best {mode} catalyst: **{best}**. It needs the lowest overpotential "
                f"(eta10 = {fmt(scored[best]['overpotential_10_mV'], 0)} mV) to deliver 10 mA/cm2. "
                "Lower overpotential means less energy wasted as heat."
            )
        else:
            lines.append(
                f"**{best}** needs an overpotential of "
                f"{fmt(scored[best]['overpotential_10_mV'], 0)} mV to deliver 10 mA/cm2. "
                "Lower overpotential means less energy wasted as heat."
            )
    fastest = max(results, key=lambda k: results[k].get("max_abs_current_mA_cm2") or 0)
    if results[fastest].get("max_abs_current_mA_cm2"):
        if multi:
            lines.append(
                f"Highest current density: **{fastest}** reaches "
                f"{fmt(results[fastest]['max_abs_current_mA_cm2'], 1)} mA/cm2, "
                "the strongest activity in this set."
            )
        else:
            lines.append(
                f"Maximum current density: **{fastest}** reaches "
                f"{fmt(results[fastest]['max_abs_current_mA_cm2'], 1)} mA/cm2."
            )
    tafel = {k: v["tafel_slope_mV_dec"] for k, v in results.items()
             if v.get("tafel_slope_mV_dec") and (v.get("max_abs_current_mA_cm2") or 0) >= 100}
    if tafel:
        tb = min(tafel, key=tafel.get)
        if multi:
            lines.append(
                f"Fastest kinetics: **{tb}** has the smallest Tafel slope "
                f"({fmt(tafel[tb], 0)} mV/dec). Its current rises fastest per volt applied."
            )
        else:
            lines.append(
                f"Tafel slope: **{tb}** has {fmt(tafel[tb], 0)} mV/dec. "
                "A smaller slope means current rises faster per volt applied."
            )
    return lines


def interpret_eis(results):
    """Plain language interpretation of EIS results, tied to the simple
    RC model. Fitted Rct and C are model dependent, not automatic
    truths, and a failed fit reports nothing."""
    if not results:
        return []
    lines = []
    multi = len(results) > 1
    fitted = {k: v for k, v in results.items()
              if v.get("fit_ok") and v.get("Rct_ohm") is not None}
    if fitted:
        best = min(fitted, key=lambda k: fitted[k]["Rct_ohm"])
        worst = max(fitted, key=lambda k: fitted[k]["Rct_ohm"])
        if multi:
            lines.append(
                f"Lowest fitted Rct under the simple RC model: **{best}** "
                f"({fmt(fitted[best]['Rct_ohm'], 1)} ohm, about "
                f"{fmt(fitted[worst]['Rct_ohm'] / fitted[best]['Rct_ohm'], 1)} times lower than {worst}). "
                "A smaller fitted semicircle points to easier charge transfer only if this circuit "
                "describes each spectrum well, so check the RMSE values."
            )
        else:
            lines.append(
                f"Fitted charge transfer resistance under the simple RC model: **{best}** has "
                f"Rct = {fmt(fitted[best]['Rct_ohm'], 1)} ohm. This is model dependent, "
                "not a direct measurement."
            )
    rs = {k: v["Rs_ohm"] for k, v in results.items() if v.get("Rs_ohm") is not None}
    if rs:
        if multi:
            lines.append(
                f"Solution resistance (Rs) is {fmt(min(rs.values()), 1)} to {fmt(max(rs.values()), 1)} ohm across samples. "
                "This comes from the electrolyte and cell setup, not the catalyst itself."
            )
        else:
            lines.append(
                f"Solution resistance (Rs) is {fmt(list(rs.values())[0], 1)} ohm. "
                "This comes from the electrolyte and cell setup, not the catalyst itself."
            )
    caps = {k: v["C_F"] for k, v in results.items() if v.get("C_F")}
    if caps:
        cb = max(caps, key=caps.get)
        if multi:
            lines.append(
                f"Largest fitted capacitance under the RC model: **{cb}** (C = {caps[cb] * 1e3:.2f} mF). "
                "A larger fitted capacitance often goes with a larger active area, but it is "
                "not automatically the double layer capacitance."
            )
        else:
            lines.append(
                f"Fitted capacitance under the RC model: **{cb}** has C = {caps[cb] * 1e3:.2f} mF. "
                "This is not automatically the double layer capacitance."
            )
    failed = [k for k, v in results.items() if not v.get("fit_ok")]
    if failed:
        lines.append(
            "No RC model fit is reported for " + ", ".join(f"**{k}**" for k in failed) +
            ". Either the simple circuit does not describe those spectra or the fit failed, "
            "so no Rct is claimed for them."
        )
    return lines


def interpret_chrono(results):
    """Plain language interpretation of chronoamperometry results."""
    if not results:
        return []
    lines = []
    multi = len(results) > 1
    ret = {k: v["retention_pct"] for k, v in results.items() if v.get("retention_pct") is not None}
    if ret:
        max_avg = max((v.get("avg_current") or 0) for v in results.values()) or 1.0
        cands = {k: r for k, r in ret.items()
                 if r <= 105 and (results[k].get("avg_current") or 0) >= 0.1 * max_avg}
        if cands:
            best = max(cands, key=lambda k: cands[k])
            r = cands[best]
            grade = "excellent" if r >= 90 else ("good" if r >= 80 else ("moderate" if r >= 65 else "poor"))
            if multi:
                lines.append(
                    f"Most stable: **{best}** keeps {fmt(r, 1)} percent of its current after "
                    f"{fmt(results[best]['duration'], 3)} hours, {grade} durability."
                )
            else:
                lines.append(
                    f"**{best}** keeps {fmt(r, 1)} percent of its current after "
                    f"{fmt(results[best]['duration'], 3)} hours, {grade} durability."
                )
        rising = [k for k, r in ret.items() if r > 105]
        if rising:
            lines.append(
                "Current increased during the test for " + ", ".join(f"**{k}**" for k in rising) +
                ". That is surface activation, not stability."
            )
    strong = max(results, key=lambda k: results[k].get("avg_current") or 0)
    if results[strong].get("avg_current"):
        if multi:
            lines.append(
                f"Highest working current: **{strong}** averages "
                f"{fmt(results[strong]['avg_current'], 2)} mA/cm2 over the whole test. "
                "The best catalyst combines high current and high retention."
            )
        else:
            lines.append(
                f"Average working current: **{strong}** averages "
                f"{fmt(results[strong]['avg_current'], 2)} mA/cm2 over the whole test."
            )
    return lines


# ============================================================================
# ML PREDICTION
# ============================================================================

_ML_FIT_FEATURES = ["Rs_ohm", "Rct_ohm", "peak_freq_Hz", "tau_s", "phase_min",
                    "Cdl_est_F", "E_onset_V", "|tafel|_mV_dec"]


def ml_section(eis_summary, lsv_summary):
    """ML prediction from fitted descriptors plus the remaining inputs.

    Models are loaded from ml_bundle.pkl next to this app file. Inputs
    already fitted from uploaded files are pre-filled; every other input
    defaults to the training median and can be edited. Scores shown are
    grouped leave-one-group-out scores (whole catalyst x condition
    groups held out), never in-sample fits.
    """
    st.divider()
    st.subheader("ML Prediction")
    st.markdown("Trained on 7 photocatalysts, 84 readings (FAU thesis dataset).")
    try:
        import joblib
    except ImportError:
        st.info("scikit-learn is not installed, so ML prediction is not "
                "available. Run the app again after installing requirements.")
        return
    bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ml_bundle.pkl")
    if not os.path.exists(bundle_path):
        st.info("ML model file ml_bundle.pkl was not found next to the app, "
                "so prediction is not available.")
        return
    try:
        bundle = joblib.load(bundle_path)
    except Exception as exc:
        st.warning(f"Could not load the ML models: {exc}")
        return

    medians = bundle["medians"]
    ranges = bundle["ranges"]
    enc = bundle["encodings"]

    auto = {}
    if eis_summary:
        for f in ["Rs_ohm", "Rct_ohm", "peak_freq_Hz", "tau_s", "phase_min",
                  "Cdl_est_F"]:
            if eis_summary.get(f) is not None:
                auto[f] = eis_summary[f]
    if lsv_summary:
        if lsv_summary.get("E_onset_V") is not None:
            auto["E_onset_V"] = lsv_summary["E_onset_V"]
        if lsv_summary.get("tafel") is not None:
            auto["|tafel|_mV_dec"] = lsv_summary["tafel"]

    def default_for(feat):
        v = auto.get(feat)
        if v is not None and np.isfinite(v):
            return float(v)
        return float(medians[feat])

    # Widget keys for fitted inputs include a signature of the fit values,
    # so the inputs refresh when new files are fitted instead of keeping
    # stale median defaults from the first render.
    import hashlib
    sig_src = "|".join(f"{k}={auto[k]:.6g}" for k in sorted(auto))
    sig = hashlib.md5(sig_src.encode()).hexdigest()[:8] if sig_src else "base"

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        metal = st.selectbox("Metal", ["Ag", "Au", "C3N4"], key="ml_metal")
    with col_b:
        loading = st.selectbox("Loading", ["ref", "0.25", "0.5", "1.0"],
                               key="ml_loading")
    with col_c:
        condition = st.selectbox("Condition",
                                 ["Dark", "410 nm", "440 nm", "Full spectrum"],
                                 key="ml_condition")
    with col_d:
        voltage = st.selectbox("EIS voltage (V)", [-0.6, -0.4, -0.2],
                               key="ml_voltage")

    st.caption("Fitted descriptors below are pre-filled from your uploaded "
               "files when available, otherwise they use training medians.")
    vals = {}
    cols = st.columns(4)
    for i, feat in enumerate(_ML_FIT_FEATURES):
        with cols[i % 4]:
            src = "from your fit" if feat in auto else "training median"
            vals[feat] = st.number_input(
                f"{feat} ({src})", value=default_for(feat),
                format="%.6g", key=f"ml_{feat}_{sig}")

    with st.expander("Other descriptors (training medians as defaults)"):
        other = [f for f in medians
                 if f not in _ML_FIT_FEATURES
                 and f not in ("metal_num", "loading_num", "condition_num",
                               "voltage")]
        ocols = st.columns(4)
        for i, feat in enumerate(other):
            with ocols[i % 4]:
                vals[feat] = st.number_input(
                    feat, value=float(medians[feat]),
                    format="%.6g", key=f"ml_{feat}")

    vals["metal_num"] = float(enc["metal"][metal])
    vals["loading_num"] = float(enc["loading"][loading])
    vals["condition_num"] = float(enc["condition"][condition])
    vals["voltage"] = float(voltage)

    if st.button("Predict with ML models", key="ml_predict"):
        out_of_range = []
        for feat in medians:
            lo, hi = ranges[feat]
            if vals[feat] < lo or vals[feat] > hi:
                out_of_range.append(feat)
        if out_of_range:
            st.warning("These inputs are outside the training range, so the "
                       "prediction is an extrapolation: "
                       + ", ".join(out_of_range))

        rows = []
        for target, tinfo in bundle["targets"].items():
            feats = tinfo["features"]
            x = np.array([[vals[f] for f in feats]], dtype=float)
            try:
                pred = float(tinfo["model"].predict(x)[0])
            except Exception as exc:
                st.error(f"Could not predict {target}: {exc}")
                continue
            s = tinfo["grouped_loocv"]
            rows.append({
                "Target": tinfo["label"],
                "Prediction": fmt(pred, 3),
                "Grouped LOOCV R2": f"{s['r2']:.3f}",
                "Grouped RMSE": fmt(s["rmse"], 3),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        for target, tinfo in bundle["targets"].items():
            drivers = tinfo.get("drivers") or []
            if not drivers:
                continue
            if tinfo["model_type"] == "lasso":
                top = drivers[:8]
                text = "; ".join(f"{d['feature']} ({d['value']:+.3f})"
                                  for d in top)
                st.info(f"Top drivers for {tinfo['label']} (LASSO "
                        f"coefficients, positive pushes the prediction up): "
                        f"{text}")
            else:
                top = drivers[:5]
                text = "; ".join(f"{d['feature']} ({d['value']:.3f})"
                                  for d in top)
                st.info(f"Top drivers for {tinfo['label']} (permutation "
                        f"importance): {text}")

        st.caption("Scores are grouped leave-one-group-out: each catalyst x "
                   "condition group was held out in turn, so these scores "
                   "are not in-sample fits.")
        j_info = bundle["targets"].get("J_m0p7")
        if j_info and j_info["grouped_loocv"]["r2"] < 0.6:
            st.warning("Photocurrent prediction is moderate (grouped R2 "
                       "below 0.6). Treat it as indicative, not precise.")


# ============================================================================
# COMPARISON DASHBOARD
# ============================================================================

def _compat_key(entry):
    return (entry["electrolyte"], entry["ph"], entry["scale"], entry["light"])


def comparison_section(candidates):
    """Comparison dashboard.

    Entries are analysed samples plus the experimental conditions they were
    measured under. Two entries are treated as comparable only when their
    electrolyte, pH, potential scale and light condition all match. Exact
    duplicates (same sample and same numbers) are refused, so uploading one
    file twice cannot move a ranking. Strict ranking leaves entries with
    missing conditions, and entries that are alone under their conditions,
    unranked.
    """
    import hashlib
    import json

    st.subheader("Comparison dashboard")
    st.write("Add analysed samples here together with their experimental "
             "conditions. A number alone never decides a ranking: entries "
             "are ranked only when their conditions match. Entries live in "
             "this session only. Restarting the app clears the list, so "
             "download the CSV to keep it. No demo data is included.")
    entries = st.session_state.setdefault("comparison_entries", [])

    if candidates:
        labels = [c["label"] for c in candidates]
        picked = st.selectbox("Sample to add", labels, key="cmp_sample")
        chosen = candidates[labels.index(picked)] if picked in labels else candidates[0]
        name = st.text_input("Catalyst name", key="cmp_name")
        electrolyte = st.text_input("Electrolyte, for example 1 M KOH",
                                    key="cmp_electrolyte")
        ph_unknown = st.checkbox("pH not recorded", key="cmp_ph_unknown")
        ph_value = st.number_input("pH", min_value=0.0, max_value=14.0,
                                   value=7.0, step=0.1, key="cmp_ph")
        light = st.selectbox("Light condition",
                             ["Not applicable", "Dark", "Light"],
                             key="cmp_light")
        if st.button("Add to comparison", key="cmp_add"):
            if not name.strip():
                st.error("Enter a catalyst name before adding.")
            else:
                metric_key = json.dumps(chosen["metrics"], sort_keys=True,
                                        default=str)
                fingerprint = hashlib.md5(
                    (chosen["sample"] + "|" + chosen["tech"] + "|" +
                     metric_key).encode("utf-8")).hexdigest()
                if any(e["fingerprint"] == fingerprint for e in entries):
                    st.error("This exact sample is already in the comparison. "
                             "A duplicate upload is not added again, so the "
                             "ranking is not affected.")
                else:
                    entry = {
                        "catalyst": name.strip(),
                        "sample": chosen["sample"],
                        "tech": chosen["tech"],
                        "mode": chosen.get("mode"),
                        "metrics": dict(chosen["metrics"]),
                        "electrolyte": (electrolyte.strip().lower()
                                        if electrolyte.strip() else None),
                        "ph": None if ph_unknown else round(float(ph_value), 1),
                        "scale": chosen["scale"],
                        "light": light,
                        "fingerprint": fingerprint,
                    }
                    name_reused = any(
                        e["catalyst"].lower() == entry["catalyst"].lower()
                        for e in entries)
                    entries.append(entry)
                    if name_reused:
                        st.warning("This catalyst name is already used with "
                                   "different data. Both entries are kept, "
                                   "but check that the name reuse is intended.")
                    st.success(f"Added {entry['catalyst']} to the comparison.")
    else:
        st.info("Upload and analyse a file first. Analysed samples can then "
                "be added here.")

    if not entries:
        return

    keys = [_compat_key(e) for e in entries]
    for entry, key in zip(entries, keys):
        missing = (entry["electrolyte"] is None or entry["ph"] is None
                   or entry["scale"] in ("Unknown (not RHE)", "Unknown"))
        entry["missing"] = missing
        entry["group_size"] = keys.count(key) if not missing else 0
        if missing:
            entry["compatibility"] = "Check: missing conditions"
        elif entry["group_size"] > 1:
            entry["compatibility"] = "Comparable"
        else:
            entry["compatibility"] = "Only entry under these conditions"

    rows = []
    for entry in entries:
        met = entry["metrics"]
        rows.append({
            "Catalyst": entry["catalyst"],
            "Sample": entry["sample"],
            "Technique": entry["tech"],
            "Mode": entry["mode"] or "Not available",
            "Onset (V)": fmt(met.get("onset"), 3),
            "eta10 (mV)": fmt(met.get("eta10"), 0),
            "eta100 (mV)": fmt(met.get("eta100"), 0),
            "Tafel (mV/dec)": fmt(met.get("tafel"), 0),
            "Rct (ohm)": fmt(met.get("rct"), 2),
            "Retention (%)": fmt(met.get("retention"), 1),
            "Potential drift (mV/h)": fmt(met.get("drift"), 1),
            "Electrolyte": entry["electrolyte"] or "Not available",
            "pH": entry["ph"] if entry["ph"] is not None else "Not available",
            "Scale": entry["scale"],
            "Light": entry["light"],
            "Compatibility": entry["compatibility"],
        })
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True)

    if len(set(keys)) > 1:
        st.warning("Entries under different conditions are listed together "
                   "but are not directly comparable. The ranking below "
                   "respects the condition groups when the strict option "
                   "is on.")
    if any(e["missing"] for e in entries):
        st.warning("At least one entry is missing electrolyte, pH or a known "
                   "potential scale. Its comparison is indicative only.")

    rank_metric = st.selectbox(
        "Ranking metric",
        ["eta10 (lower is better)", "eta100 (lower is better)",
         "Tafel absolute value (lower is better)", "Rct (lower is better)",
         "Retention (higher is better)", "Max |j| (higher is better)",
         "Potential drift absolute (lower is better)"],
        key="cmp_rank_metric")
    strict = st.checkbox("Rank only entries with identical conditions",
                         value=True, key="cmp_strict")
    metric_map = {"eta10 (lower is better)": ("eta10", False),
                  "eta100 (lower is better)": ("eta100", False),
                  "Tafel absolute value (lower is better)": ("tafel_abs", False),
                  "Rct (lower is better)": ("rct", False),
                  "Retention (higher is better)": ("retention", True),
                  "Max |j| (higher is better)": ("max_j", True),
                  "Potential drift absolute (lower is better)": ("drift_abs", False)}
    mkey, higher = metric_map[rank_metric]

    def rank_value(entry):
        key = mkey[:-4] if mkey.endswith("_abs") else mkey
        value = entry["metrics"].get(key)
        if value is None:
            return None
        return abs(value) if mkey.endswith("_abs") else value

    ranked = []
    for entry in entries:
        value = rank_value(entry)
        if value is None:
            continue
        if strict and len(entries) > 1 and (entry["missing"] or entry["group_size"] < 2):
            continue
        ranked.append((entry, value))
    ranked.sort(key=lambda pair: pair[1], reverse=higher)
    if ranked:
        st.write("Ranking:")
        for pos, (entry, value) in enumerate(ranked, start=1):
            st.write(f"{pos}. {entry['catalyst']} ({entry['sample']}): "
                     f"{fmt(value, 2)}")
    else:
        st.info("No entry can be ranked under the current setting. Switch off "
                "the strict option to rank across conditions, or add entries "
                "with matching conditions and this metric.")
    if strict:
        st.caption("Strict ranking hides entries with missing conditions and "
                   "entries that are alone under their conditions.")

    st.download_button("Download comparison as CSV",
                       table.to_csv(index=False).encode("utf-8"),
                       file_name="comparison.csv", mime="text/csv",
                       key="cmp_download")
    labels = [f"{e['catalyst']} ({e['sample']})" for e in entries]
    remove = st.multiselect("Remove entries", labels, key="cmp_remove")
    if st.button("Remove selected", key="cmp_remove_btn") and remove:
        st.session_state["comparison_entries"] = [
            e for e, label in zip(entries, labels) if label not in remove]
        st.success("Selected entries removed.")
    if st.button("Clear comparison", key="cmp_clear"):
        st.session_state["comparison_entries"] = []
        st.success("Comparison cleared.")


# ============================================================================
# PHOTOELECTROCHEMICAL (PEC) INTERPRETATION
# ============================================================================

def pec_figure(v_grid, j_photo, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(v_grid, j_photo, color=color, linewidth=2)
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Photocurrent density (mA/cm2)")
    ax.set_title("Photocurrent: light minus dark")
    ax.grid(True, alpha=0.3)
    return fig


def pec_lsv_section(lsv_entries):
    """Photoelectrochemical interpretation from a dark and a light LSV.

    Photocurrent is the light current minus the dark current at the same
    potential. ABPE is calculated only when the user enters a measured
    incident power density, and only for anodic photocurrent. Photocurrent
    from a three electrode measurement is not solar to hydrogen
    efficiency, which needs a two electrode device measurement with
    measured gas production.
    """
    st.subheader("Photoelectrochemical (PEC) interpretation")
    st.write("This section interprets photoelectrochemical data. Upload a "
             "dark LSV and a light LSV of the same sample, then pick them "
             "below. Photocurrent is the light current minus the dark "
             "current at the same potential. A photocurrent value alone is "
             "not a device efficiency.")
    if len(lsv_entries) < 2:
        st.info("Upload two LSV files of the same sample, one measured in "
                "the dark and one under illumination, to see photocurrent, "
                "onset shift and ABPE here.")
        return
    labels = [f"{e['name']} ({e['m'].get('mode', 'LSV')})" for e in lsv_entries]
    dark_label = st.selectbox("Dark LSV sample", labels, key="pec_dark")
    light_label = st.selectbox("Light LSV sample", labels, key="pec_light",
                               index=1 if len(labels) > 1 else 0)
    dark = lsv_entries[labels.index(dark_label)] if dark_label in labels else lsv_entries[0]
    light = lsv_entries[labels.index(light_label)] if light_label in labels else lsv_entries[0]
    if dark is light:
        st.error("Choose two different samples for dark and light.")
        return
    power = st.number_input("Incident power density (mW/cm2)",
                            min_value=0.0, value=0.0, step=10.0, key="pec_power")
    st.caption("Enter 0 when the lamp power was not measured. ABPE is then "
               "not calculated. Enter a calibrated value, for example 100 "
               "for 1 sun, to see ABPE.")
    try:
        v_dark = dark["data"]["Potential_V"].to_numpy(dtype=float)
        j_dark = dark["data"]["Current_Density"].to_numpy(dtype=float)
        v_light = light["data"]["Potential_V"].to_numpy(dtype=float)
        j_light = light["data"]["Current_Density"].to_numpy(dtype=float)
    except Exception as exc:
        st.error(f"Could not read the selected curves: {exc}")
        return
    lo = max(float(np.min(v_dark)), float(np.min(v_light)))
    hi = min(float(np.max(v_dark)), float(np.max(v_light)))
    if hi <= lo:
        st.error("The two curves do not share a potential range, so no "
                 "photocurrent can be calculated.")
        return
    grid = np.linspace(lo, hi, 500)
    order_d = np.argsort(v_dark)
    order_l = np.argsort(v_light)
    j_d = np.interp(grid, v_dark[order_d], j_dark[order_d])
    j_l = np.interp(grid, v_light[order_l], j_light[order_l])
    j_photo = j_l - j_d
    both_rhe = dark["scale"].startswith("RHE") and light["scale"].startswith("RHE")
    idx_max = int(np.argmax(np.abs(j_photo)))
    max_photo = float(j_photo[idx_max])
    v_at_max = float(grid[idx_max])
    j_at_123 = float(np.interp(1.23, grid, j_photo)) if (both_rhe and lo <= 1.23 <= hi) else None
    onset_dark = dark["m"].get("onset_potential_V")
    onset_light = light["m"].get("onset_potential_V")
    shift = (onset_light - onset_dark) if (onset_dark is not None and onset_light is not None) else None
    abpe_max = None
    v_abpe = None
    if power > 0 and both_rhe and max_photo > 0:
        valid = (grid < 1.23) & (j_photo > 0)
        if np.any(valid):
            abpe = j_photo[valid] * (1.23 - grid[valid]) / power * 100.0
            k = int(np.argmax(abpe))
            abpe_max = float(abpe[k])
            v_abpe = float(grid[valid][k])

    st.write(f"Photocurrent at 1.23 V vs RHE: {fmt(j_at_123, 2)} mA/cm2")
    if j_at_123 is None:
        st.info("The 1.23 V value needs both curves on the RHE scale and a "
                "shared range covering 1.23 V.")
    st.write(f"Maximum photocurrent: {fmt(max_photo, 2)} mA/cm2 at "
             f"{fmt(v_at_max, 3)} V")
    if shift is None:
        st.write("Onset shift under illumination: Not available")
    elif shift < 0:
        st.write(f"Onset shift under illumination: the light onset is "
                 f"{fmt(abs(shift) * 1000.0, 0)} mV lower than the dark onset.")
    elif shift > 0:
        st.write(f"Onset shift under illumination: the light onset is "
                 f"{fmt(shift * 1000.0, 0)} mV higher than the dark onset.")
    else:
        st.write("Onset shift under illumination: no shift detected.")
    if abpe_max is not None:
        st.write(f"Maximum ABPE: {fmt(abpe_max, 2)} percent at "
                 f"{fmt(v_abpe, 3)} V")
    elif power == 0:
        st.info("ABPE is not calculated because no incident power was entered.")
    elif not both_rhe:
        st.info("ABPE needs both curves on the RHE scale.")
    elif max_photo <= 0:
        st.info("ABPE is not calculated here because the photocurrent is "
                "cathodic. The ABPE form used here applies to anodic "
                "photocurrent.")
    else:
        st.info("ABPE is not available in the shared potential range.")
    if max_photo > 0:
        st.info("The photocurrent is anodic: under illumination the sample "
                "behaves as a photoanode.")
    elif max_photo < 0:
        st.info("The photocurrent is cathodic: under illumination the sample "
                "behaves as a photocathode.")
    else:
        st.info("No photocurrent is detected in the shared range.")
    if dark["m"].get("mode") != light["m"].get("mode"):
        st.warning("The two samples have different modes, so treat this "
                   "comparison with care.")
    st.info("Photocurrent from a three electrode cell is not solar to "
            "hydrogen efficiency. STH needs a two electrode device "
            "measurement with measured gas production.")
    fig = pec_figure(grid, j_photo, "#1f77b4")
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("Download photocurrent figure as PNG",
                       figure_bytes(pec_figure(grid, j_photo, "#1f77b4"), "png", 300),
                       file_name="photocurrent.png", mime="image/png",
                       key="pec_png")
    st.download_button("Download photocurrent figure as TIF",
                       figure_bytes(pec_figure(grid, j_photo, "#1f77b4"), "tiff", 300),
                       file_name="photocurrent.tif", mime="image/tiff",
                       key="pec_tif")
    plt.close("all")


# ============================================================================
# TAFEL FIT INSPECTOR
# ============================================================================

def fit_tafel_interval(v, j, j_min=None, j_max=None):
    """Tafel fit over a chosen |j| interval, with points and residuals.

    Same convention as tafel_slope: log10|j| is fitted against potential,
    and the reported slope is 1000 divided by the absolute fitted slope,
    in mV per decade. Returns None when too few points remain."""
    v = np.asarray(v, dtype=float)
    aj = np.abs(np.asarray(j, dtype=float))
    mask = (aj > 0) & np.isfinite(v) & np.isfinite(aj)
    v, aj = v[mask], aj[mask]
    if len(v) < 10:
        return None
    logj = np.log10(aj)
    if j_min is not None and j_max is not None and 0 < j_min < j_max:
        sel = (aj >= j_min) & (aj <= j_max)
        lo_j, hi_j = float(j_min), float(j_max)
    else:
        lo, hi = np.quantile(logj, 0.15), np.quantile(logj, 0.75)
        sel = (logj >= lo) & (logj <= hi)
        lo_j, hi_j = float(10 ** lo), float(10 ** hi)
    if sel.sum() < 5:
        return None
    slope, intercept = np.polyfit(v[sel], logj[sel], 1)
    if abs(slope) < 1e-9:
        return None
    pred = slope * v[sel] + intercept
    ss_res = float(np.sum((logj[sel] - pred) ** 2))
    ss_tot = float(np.sum((logj[sel] - np.mean(logj[sel])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"slope_mV_dec": float(1000.0 / abs(slope)),
            "intercept": float(intercept), "slope_per_dec": float(slope),
            "r2": r2, "points": int(sel.sum()), "j_min": lo_j, "j_max": hi_j,
            "logj": logj[sel], "v": v[sel], "resid": logj[sel] - pred}


def tafel_inspector_figure(fit):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(fit["logj"], fit["v"], "o", color="#1f77b4", markersize=4)
    xs = np.linspace(float(np.min(fit["logj"])), float(np.max(fit["logj"])), 100)
    ys = (xs - fit["intercept"]) / fit["slope_per_dec"]
    ax1.plot(xs, ys, color="#d62728", linewidth=2)
    ax1.set_ylabel("Potential (V)")
    ax1.set_title("Tafel fit: points and fitted line")
    ax1.grid(True, alpha=0.3)
    ax2.plot(fit["logj"], fit["resid"], "o", color="#1f77b4", markersize=4)
    ax2.axhline(0, color="#555555", linewidth=1)
    ax2.set_xlabel("log10 |j| (j in mA/cm2)")
    ax2.set_ylabel("Residual (log10 units)")
    ax2.set_title("Residuals: measured minus fitted, in log10 units")
    ax2.grid(True, alpha=0.3)
    return fig


# ============================================================================
# CPE CIRCUIT FIT FOR EIS
# ============================================================================

def _cpe_impedance(freq, rs, rct, q, n):
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    return rs + 1.0 / (1.0 / rct + q * (1j * w) ** n)


def fit_rcpe(freq, zr, zi, rs0, rct0, c0):
    """CPE circuit fit: Rs + (Rct || CPE). Returns None when SciPy is
    missing, when too few points exist, or when the fit fails. Nothing
    is substituted."""
    if (not _HAVE_SCIPY or len(freq) < 8 or len(freq) != len(zr)
            or len(freq) != len(zi)):
        return None
    freq = np.asarray(freq, dtype=float)
    zr = np.asarray(zr, dtype=float)
    zi = np.asarray(zi, dtype=float)
    if not (np.all(freq > 0) and np.all(np.isfinite(freq))
            and np.all(np.isfinite(zr)) and np.all(np.isfinite(zi))):
        return None
    z = zr + 1j * zi

    def resid(p):
        rs, rct, q, n = p
        fit = _cpe_impedance(freq, rs, rct, q, n)
        return np.concatenate([(fit.real - zr), (fit.imag - zi)])

    try:
        sol = least_squares(resid, [rs0, rct0, max(c0, 1e-9), 0.9],
                            bounds=([0, 0, 1e-12, 0.3],
                                    [np.inf, np.inf, np.inf, 1.0]))
        rs, rct, q, n = [float(x) for x in sol.x]
        if not all(np.isfinite([rs, rct, q, n])) or rct <= 0 or q <= 0:
            return None
        fit = _cpe_impedance(freq, rs, rct, q, n)
        rmse = float(np.sqrt(np.mean(np.abs(fit - z) ** 2)))
        return {"Rs_ohm": rs, "Rct_ohm": rct, "Q": q, "n": n,
                "RMSE_ohm": rmse, "fit": fit}
    except Exception:
        return None


def cpe_nyquist_figure(data, fit):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["Zreal"], -data["Zimag"], "o", color="#1f77b4", markersize=4)
    ax.plot(fit.real, -fit.imag, color="#d62728", linewidth=2)
    ax.set_xlabel("Z' (ohm)")
    ax.set_ylabel("Z'' (ohm)")
    ax.set_title("Nyquist: measured points and CPE fit")
    ax.grid(True, alpha=0.3)
    return fig


def cpe_bode_figure(freq, zr, zi, fit):
    order = np.argsort(freq)
    f = freq[order]
    zmag = np.sqrt(zr[order] ** 2 + zi[order] ** 2)
    phase = np.degrees(np.arctan2(zi[order], zr[order]))
    fmag = np.abs(fit[order])
    fphase = np.degrees(np.arctan2(fit[order].imag, fit[order].real))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(f, zmag, "o", color="#1f77b4", markersize=4)
    ax1.plot(f, fmag, color="#d62728", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_ylabel("|Z| (ohm)")
    ax1.set_title("Bode magnitude: measured points and CPE fit")
    ax1.grid(True, alpha=0.3)
    ax2.plot(f, phase, "o", color="#1f77b4", markersize=4)
    ax2.plot(f, fphase, color="#d62728", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title("Bode phase: measured points and CPE fit")
    ax2.grid(True, alpha=0.3)
    return fig


def cpe_resid_figure(freq, zr, zi, fit):
    order = np.argsort(freq)
    f = freq[order]
    r_re = fit[order].real - zr[order]
    r_im = fit[order].imag - zi[order]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(f, r_re, "o", color="#1f77b4", markersize=4)
    ax1.axhline(0, color="#555555", linewidth=1)
    ax1.set_xscale("log")
    ax1.set_ylabel("Real residual (ohm)")
    ax1.set_title("CPE fit residuals against frequency")
    ax1.grid(True, alpha=0.3)
    ax2.plot(f, r_im, "o", color="#1f77b4", markersize=4)
    ax2.axhline(0, color="#555555", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Imaginary residual (ohm)")
    ax2.grid(True, alpha=0.3)
    return fig


# ============================================================================
# CHRONOPOTENTIOMETRY
# ============================================================================

def parse_cp(df, current_unit="auto", area_cm2=0.5):
    """Chronopotentiometry: constant current, potential against time.
    Time is returned in hours: seconds divided by 3600 when the column
    is T(s). Current density uses to_current_density, same rule as
    chronoamperometry."""
    tcol = None
    convert_time = True
    for c in df.columns:
        if str(c).strip().lower() in ("t(s)", "t (s)"):
            tcol = c
            break
    if tcol is None:
        for c in df.columns:
            if str(c).strip().lower() in ("time", "time (s)"):
                tcol = c
                convert_time = False
                break
    if tcol is None:
        tcol = pick_column(df, ["time", "duration", "t ("], 0)
        convert_time = False
    vcol = pick_column(df, ["e (v", "e(v", "potential", "volt", "v vs"], 0)
    jcol = pick_column(df, ["i (", "i(", "current", "dens", "j (", "j("], 1)
    j = numeric_series(df, jcol)
    unit = resolve_current_unit(jcol, current_unit)
    j = to_current_density(j.to_numpy(dtype=float), unit, area_cm2)
    v = numeric_series(df, vcol)
    if "mv" in str(vcol).lower():
        v = v / 1000.0
    t = numeric_series(df, tcol)
    if convert_time:
        t = t / 3600.0
    out = pd.DataFrame({"Time": t, "Potential_V": v,
                        "Current_Density": j}).dropna()
    return out.sort_values("Time").reset_index(drop=True)


def analyze_cp(data):
    t = data["Time"].to_numpy(dtype=float)
    v = data["Potential_V"].to_numpy(dtype=float)
    j = data["Current_Density"].to_numpy(dtype=float)
    n = len(t)
    k = max(1, n // 20)
    v_start = float(np.mean(v[:k]))
    v_end = float(np.mean(v[-k:]))
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    drift = (v_end - v_start) / duration * 1000.0 if duration > 0 else None
    return {"initial_potential": v_start, "final_potential": v_end,
            "mean_potential": float(np.mean(v)) if n else None,
            "drift_mV_h": drift,
            "applied_current": float(np.mean(j)) if n else None,
            "duration": duration, "points": int(n)}


def cp_figure(data, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["Time"], data["Potential_V"], color=color, linewidth=2)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Potential (V)")
    ax.set_title("Chronopotentiometry: potential at constant current")
    ax.grid(True, alpha=0.3)
    return fig


# ============================================================================
# TRANSIENT PHOTOCURRENT AND IPCE
# ============================================================================

def parse_intervals(text):
    """Light on intervals in hours, written as 0 to 0.05, 0.1 to 0.15.
    A hyphen is also accepted as a separator. Returns a list of
    (start, end) pairs, or None when the text is invalid."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if " to " in part:
            a, b = part.split(" to ", 1)
        elif "-" in part:
            a, b = part.split("-", 1)
        else:
            return None
        try:
            lo, hi = float(a.strip()), float(b.strip())
        except ValueError:
            return None
        if not hi > lo:
            return None
        out.append((lo, hi))
    return out or None


def transient_stats(t, j, intervals):
    """Per cycle light on and light off means from a chrono curve.

    The dark baseline is the mean current outside all light on
    intervals. Photocurrent per cycle is the on mean minus that
    baseline. Response time is the time from the first interval start
    until the current first reaches 90 percent of the step, in
    seconds. Signed current is used, so an anodic step gives a
    positive photocurrent."""
    t = np.asarray(t, dtype=float)
    j = np.asarray(j, dtype=float)
    order = np.argsort(t)
    t, j = t[order], j[order]
    on_mask = np.zeros(len(t), dtype=bool)
    for lo, hi in intervals:
        on_mask |= (t >= lo) & (t <= hi)
    off_mask = ~on_mask
    if on_mask.sum() < 2 or off_mask.sum() < 2:
        return None
    off_mean = float(np.mean(j[off_mask]))
    cycles = []
    for lo, hi in intervals:
        m = (t >= lo) & (t <= hi)
        if m.sum() < 2:
            continue
        on_mean = float(np.mean(j[m]))
        cycles.append({"start_h": lo, "end_h": hi, "on_mean": on_mean,
                       "photo": on_mean - off_mean})
    if not cycles:
        return None
    photos = np.array([c["photo"] for c in cycles])
    mean_photo = float(np.mean(photos))
    stability = (float(np.std(photos) / abs(mean_photo) * 100.0)
                 if mean_photo != 0 else None)
    lo0, hi0 = intervals[0]
    m0 = (t >= lo0) & (t <= hi0)
    t0, j0 = t[m0], j[m0]
    on_mean0 = cycles[0]["on_mean"]
    target = off_mean + 0.9 * (on_mean0 - off_mean)
    if on_mean0 > off_mean:
        hit = np.where(j0 >= target)[0]
    elif on_mean0 < off_mean:
        hit = np.where(j0 <= target)[0]
    else:
        hit = np.array([])
    t90 = float((t0[hit[0]] - lo0) * 3600.0) if len(hit) else None
    return {"off_mean": off_mean, "cycles": cycles,
            "mean_photo": mean_photo, "stability_pct": stability,
            "t90_s": t90}


def transient_figure(t, j, intervals, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, j, color=color, linewidth=2)
    for lo, hi in intervals:
        ax.axvspan(lo, hi, color="#ff7f0e", alpha=0.2)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Current density (mA/cm2)")
    ax.set_title("Transient photocurrent: shaded regions are light on")
    ax.grid(True, alpha=0.3)
    return fig


def ipce_values(wl, j, power):
    """IPCE in percent: 1240 times photocurrent magnitude divided by
    wavelength times incident power, times 100. Photocurrent in
    mA/cm2, power in mW/cm2, wavelength in nm. Magnitude is used, so
    a signed photocurrent still gives a positive IPCE."""
    wl = np.asarray(wl, dtype=float)
    j = np.asarray(j, dtype=float)
    p = np.asarray(power, dtype=float)
    mask = (np.isfinite(wl) & np.isfinite(j) & np.isfinite(p)
            & (wl > 0) & (p > 0))
    wl, j, p = wl[mask], j[mask], p[mask]
    if len(wl) < 2:
        return None
    ipce = 1240.0 * np.abs(j) / (wl * p) * 100.0
    return {"wl": wl, "j": j, "power": p, "ipce": ipce}


def ipce_figure(wl, ipce, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wl, ipce, color=color, linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("IPCE (percent)")
    ax.set_title("IPCE against wavelength")
    ax.grid(True, alpha=0.3)
    return fig


def pec_section(lsv_entries, eis_entries=None, chrono_entries=None):
    """Photoelectrochemical analyses beyond dark and light LSV: charge
    transfer under illumination from dark and light EIS, transient
    photocurrent from light on and off intervals in a chrono curve,
    and IPCE from wavelength dependent photocurrent and power."""
    eis_entries = eis_entries or []
    chrono_entries = chrono_entries or []
    pec_lsv_section(lsv_entries)

    st.subheader("Charge transfer under illumination (PEIS)")
    fitted = [e for e in eis_entries if e["m"].get("fit_ok")]
    if len(fitted) < 2:
        st.info("Upload two EIS spectra with successful RC fits, one dark "
                "and one illuminated, to compare charge transfer resistance "
                "under illumination here.")
    else:
        labels = [e["name"] for e in fitted]
        dark_label = st.selectbox("Dark EIS sample", labels, key="peis_dark")
        light_label = st.selectbox("Light EIS sample", labels, key="peis_light",
                                   index=1 if len(labels) > 1 else 0)
        dark = fitted[labels.index(dark_label)] if dark_label in labels else fitted[0]
        light = fitted[labels.index(light_label)] if light_label in labels else fitted[0]
        if dark is light:
            st.error("Choose two different spectra for dark and light.")
        else:
            r_dark = dark["m"]["Rct_ohm"]
            r_light = light["m"]["Rct_ohm"]
            change = (r_light - r_dark) / r_dark * 100.0 if r_dark else None
            st.write(f"Rct dark (RC model): {fmt(r_dark, 2)} ohm")
            st.write(f"Rct light (RC model): {fmt(r_light, 2)} ohm")
            st.write(f"Change under illumination: {fmt(change, 1)} percent")
            if change is not None and change < 0:
                st.info("Rct falls under illumination, which suggests faster "
                        "charge transfer in the light. It does not by itself "
                        "prove higher device efficiency.")
            elif change is not None and change > 0:
                st.info("Rct rises under illumination here, which suggests "
                        "slower charge transfer in the light for this sample.")
            else:
                st.info("No change in Rct is detected under illumination.")

    st.subheader("Transient photocurrent (light on and off)")
    if not chrono_entries:
        st.info("Upload a chronoamperometry file recorded with light "
                "switching, then enter the light on intervals below.")
    else:
        labels = [e["name"] for e in chrono_entries]
        picked = st.selectbox("Chrono sample", labels, key="trans_sample")
        entry = chrono_entries[labels.index(picked)] if picked in labels else chrono_entries[0]
        text = st.text_input("Light on intervals (h), for example "
                             "0 to 0.05, 0.1 to 0.15", key="trans_intervals")
        intervals = parse_intervals(text) if text.strip() else None
        if text.strip() and intervals is None:
            st.error("Intervals could not be read. Write them as start to "
                     "end in hours, separated by commas.")
        elif intervals:
            stats = transient_stats(entry["data"]["Time"].to_numpy(dtype=float),
                                    entry["data"]["Current_Density"].to_numpy(dtype=float),
                                    intervals)
            if stats is None:
                st.error("Not enough points inside and outside the light on "
                         "intervals to analyse.")
            else:
                rows = [{"Interval start (h)": c["start_h"],
                         "Interval end (h)": c["end_h"],
                         "Light on mean (mA/cm2)": fmt(c["on_mean"], 2),
                         "Photocurrent (mA/cm2)": fmt(c["photo"], 2)}
                        for c in stats["cycles"]]
                table = pd.DataFrame(rows)
                st.dataframe(table, use_container_width=True)
                st.write(f"Dark baseline: {fmt(stats['off_mean'], 2)} mA/cm2")
                st.write(f"Mean photocurrent: {fmt(stats['mean_photo'], 2)} mA/cm2")
                if stats["stability_pct"] is not None:
                    st.write(f"Cycle to cycle variation: {fmt(stats['stability_pct'], 1)} percent")
                if stats["t90_s"] is not None:
                    st.write(f"Response time to 90 percent of the first step: {fmt(stats['t90_s'], 1)} s")
                else:
                    st.write("Response time: the current does not reach 90 percent of the first step inside the first interval.")
                t_arr = entry["data"]["Time"].to_numpy(dtype=float)
                j_arr = entry["data"]["Current_Density"].to_numpy(dtype=float)
                fig = transient_figure(t_arr, j_arr, intervals, "#1f77b4")
                st.pyplot(fig)
                plt.close(fig)
                base = safe_filename(entry["name"])
                st.download_button("Download transient figure as PNG",
                                   figure_bytes(transient_figure(t_arr, j_arr, intervals, "#1f77b4"), "png", 300),
                                   file_name=f"{base}_transient.png", mime="image/png",
                                   key="trans_png")
                st.download_button("Download transient figure as TIF",
                                   figure_bytes(transient_figure(t_arr, j_arr, intervals, "#1f77b4"), "tiff", 300),
                                   file_name=f"{base}_transient.tif", mime="image/tiff",
                                   key="trans_tif")
                plt.close("all")

    st.subheader("IPCE")
    st.write("Upload a file with wavelength, photocurrent and incident "
             "power columns. IPCE is calculated per wavelength from "
             "measured power only, never from an assumed spectrum.")
    ipce_files = st.file_uploader("Upload IPCE data files", type=["csv", "txt", "xlsx"],
                                  accept_multiple_files=True, key="ipce_files")
    for f in ipce_files or []:
        try:
            tables = read_upload(f)
        except Exception as exc:
            st.error(f"Could not read {f.name}: {exc}")
            continue
        for name, df, cs_id in tables:
            def find_col(keywords):
                for kw in keywords:
                    for c in df.columns:
                        if kw in str(c).lower():
                            return c
                return None
            wl_col = find_col(["wavelength", "lambda", "nm"])
            j_col = find_col(["photocurrent", "current", "dens"])
            p_col = find_col(["power", "irradiance", "intensity"])
            if wl_col is None or j_col is None:
                st.error(f"{name}: wavelength and photocurrent columns were "
                         "not found. The file needs a wavelength column and "
                         "a photocurrent column.")
                continue
            wl = numeric_series(df, wl_col).to_numpy(dtype=float)
            jcur = numeric_series(df, j_col).to_numpy(dtype=float)
            if p_col is not None:
                pwr = numeric_series(df, p_col).to_numpy(dtype=float)
            else:
                const_p = st.number_input(f"Constant power for {name} (mW/cm2)",
                                          min_value=0.0, value=0.0, step=1.0,
                                          key=f"ipce_power_{safe_filename(name)}")
                if const_p <= 0:
                    st.info(f"{name}: enter a constant incident power, or add "
                            "a power column to the file.")
                    continue
                pwr = np.full(len(wl), const_p)
            res = ipce_values(wl, jcur, pwr)
            if res is None:
                st.error(f"{name}: not enough valid rows to calculate IPCE.")
                continue
            rows = [{"Wavelength (nm)": fmt(w, 1),
                     "Photocurrent (mA/cm2)": fmt(v, 3),
                     "Power (mW/cm2)": fmt(p, 3),
                     "IPCE (percent)": fmt(e, 2)}
                    for w, v, p, e in zip(res["wl"], res["j"], res["power"], res["ipce"])]
            table = pd.DataFrame(rows)
            st.dataframe(table, use_container_width=True)
            k = int(np.argmax(res["ipce"]))
            st.write(f"Maximum IPCE: {fmt(res['ipce'][k], 2)} percent at {fmt(res['wl'][k], 1)} nm")
            if np.any(res["ipce"] > 100):
                st.warning("At least one IPCE value is above 100 percent, which is unphysical. Check the power calibration and the current units.")
            fig = ipce_figure(res["wl"], res["ipce"], "#1f77b4")
            st.pyplot(fig)
            plt.close(fig)
            base = safe_filename(name)
            st.download_button("Download IPCE table as CSV",
                               table.to_csv(index=False).encode("utf-8"),
                               file_name=f"{base}_ipce.csv", mime="text/csv",
                               key=f"ipce_csv_{base}")
            st.download_button("Download IPCE figure as PNG",
                               figure_bytes(ipce_figure(res["wl"], res["ipce"], "#1f77b4"), "png", 300),
                               file_name=f"{base}_ipce.png", mime="image/png",
                               key=f"ipce_png_{base}")
            plt.close("all")


# ============================================================================
# PRESENTATION UI (v13: styling only, analysis untouched)
# ============================================================================

APP_CSS = """
<style>
.stButton > button, .stDownloadButton > button {
    background-color: #0E7C7B !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.45rem 1.1rem !important;
    font-weight: 600 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #0A5E5D !important;
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"] section {
    border: 2px dashed #0E7C7B !important;
    border-radius: 14px !important;
}
[data-testid="stToolbar"] { display: none !important; }
footer { display: none !important; }
</style>
"""


def render_hero():
    """Top banner for presentation. Text only, no analysis here."""
    st.markdown("""
<div style="background:linear-gradient(135deg,#0B3C5D 0%,#0E7C7B 100%);padding:28px 32px;border-radius:18px;color:#FFFFFF;">
<div style="font-size:42px;font-weight:800;letter-spacing:-0.5px;">⚡ ElectroCatalyst AI</div>
<div style="font-size:17px;opacity:0.92;margin-top:6px;">Real data only. Upload your own instrument files. Every number below is computed from the files you upload. No demo data, no sample data. The app detects the technique from the data itself, so you do not select anything.</div>
<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;">
<span style="background:rgba(255,255,255,0.16);padding:6px 12px;border-radius:999px;font-size:13px;">LSV</span>
<span style="background:rgba(255,255,255,0.16);padding:6px 12px;border-radius:999px;font-size:13px;">EIS</span>
<span style="background:rgba(255,255,255,0.16);padding:6px 12px;border-radius:999px;font-size:13px;">Chronoamperometry</span>
<span style="background:rgba(255,255,255,0.16);padding:6px 12px;border-radius:999px;font-size:13px;">Chronopotentiometry</span>
<span style="background:rgba(255,255,255,0.16);padding:6px 12px;border-radius:999px;font-size:13px;">PEC</span>
<span style="background:rgba(255,255,255,0.16);padding:6px 12px;border-radius:999px;font-size:13px;">ML Prediction</span>
<span style="background:rgba(255,255,255,0.16);padding:6px 12px;border-radius:999px;font-size:13px;">Comparison</span>
</div>
<div style="display:flex;gap:10px;margin-top:18px;">
<div style="flex:1;background:rgba(255,255,255,0.12);border-radius:12px;padding:10px 12px;"><b>1. Upload</b><br><span style="opacity:0.85;font-size:13px;">Instrument files</span></div>
<div style="flex:1;background:rgba(255,255,255,0.12);border-radius:12px;padding:10px 12px;"><b>2. Detect</b><br><span style="opacity:0.85;font-size:13px;">Technique from data</span></div>
<div style="flex:1;background:rgba(255,255,255,0.12);border-radius:12px;padding:10px 12px;"><b>3. Analyse</b><br><span style="opacity:0.85;font-size:13px;">Fits and parameters</span></div>
<div style="flex:1;background:rgba(255,255,255,0.12);border-radius:12px;padding:10px 12px;"><b>4. Present</b><br><span style="opacity:0.85;font-size:13px;">Plots, tables, comparison</span></div>
</div>
</div>
""", unsafe_allow_html=True)


def metric_cards(items):
    """Headline numbers as cards. Items are (label, value) pairs; any
    value that is not available is left out. Display only."""
    items = [(lab, val) for lab, val in items
             if val is not None and val != "Not available"]
    if not items:
        return
    cols = st.columns(len(items))
    for col, (lab, val) in zip(cols, items):
        with col:
            st.markdown(
                f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
                f'border-top:4px solid #0E7C7B;border-radius:14px;'
                f'padding:14px 16px;box-shadow:0 2px 8px rgba(11,60,93,0.06);">'
                f'<div style="font-size:12px;letter-spacing:0.6px;'
                f'text-transform:uppercase;color:#64748B;font-weight:700;">'
                f'{lab}</div>'
                f'<div style="font-size:26px;font-weight:800;color:#0B3C5D;'
                f'margin-top:4px;">{val}</div></div>',
                unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="ElectroCatalyst AI", page_icon="⚡", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    render_hero()

    st.sidebar.markdown("### ⚡ Controls")
    st.sidebar.caption("Everything below changes the analysis live.")
    dpi = st.sidebar.selectbox("Figure download quality (DPI)", [150, 300, 600], index=1)
    area_cm2 = st.sidebar.number_input("Electrode area (cm2)", min_value=0.01, value=0.5, step=0.05)
    st.sidebar.caption("Used only when the current itself is in A or mA. A density column is never divided by area again.")
    unit_choice = st.sidebar.selectbox(
        "Current unit in uploaded files",
        ["Auto (from column header)", "mA/cm²", "A/cm²", "mA", "A"],
    )
    current_unit = "auto" if unit_choice.startswith("Auto") else unit_choice
    st.sidebar.caption("Auto reads the unit from the column header. Choose mA/cm² when the instrument already gives current density.")
    potential_scale = st.sidebar.selectbox(
        "Potential reference scale",
        ["Unknown", "RHE", "Other - known RHE offset"],
    )
    rhe_offset = None
    if potential_scale == "RHE":
        rhe_offset = 0.0
    elif potential_scale == "Other - known RHE offset":
        rhe_offset = st.sidebar.number_input("Verified offset to RHE (V)", value=0.0, step=0.01, format="%.4f")
    if potential_scale == "Unknown":
        st.sidebar.caption("Unknown means overpotential is not calculated. Choose RHE or enter a verified offset to see eta values.")
    tafel_min = st.sidebar.number_input("Tafel fit min (mA/cm2)", min_value=0.0, value=0.0, step=1.0)
    tafel_max = st.sidebar.number_input("Tafel fit max (mA/cm2)", min_value=0.0, value=0.0, step=1.0)
    st.sidebar.caption("0 and 0 means the automatic band is used for the Tafel fit.")

    files = st.file_uploader(
        "Upload your files, one file per sample. Excel files are read sheet by sheet.",
        type=["csv", "txt", "xlsx"], accept_multiple_files=True,
    )
    if not files:
        st.info("Upload a file to begin. Supported techniques: LSV, EIS, "
                "Chronoamperometry, Chronopotentiometry. Other techniques "
                "are reported as not supported and are not plotted.")

    samples = []
    for f in files:
        try:
            tables = read_upload(f)
        except Exception as exc:
            st.error(f"Could not read {f.name}: {exc}")
            continue
        if not tables:
            st.error(f"Could not read {f.name}: no data table found.")
            continue
        for name, df, cs_id in tables:
            tech = detect_technique(df, cs_id)
            samples.append({"name": name, "df": df, "tech": tech})

    if samples:
        st.subheader("Detected techniques")
        for s in samples:
            st.write(f"{s['name']}: detected as {s['tech']}")

    eis_summary = None
    lsv_summary = None
    candidates = []
    lsv_entries = []
    eis_entries = []
    chrono_entries = []

    supported = [s for s in samples if s["tech"] in ("LSV", "EIS", "Chrono", "CP")]
    unsupported = [s for s in samples if s["tech"] not in ("LSV", "EIS", "Chrono")]

    if unsupported:
        st.subheader("Not supported")
        for s in unsupported:
            st.warning(
                f"{s['name']}: detected as {s['tech']}. This app supports LSV, "
                "EIS, Chronoamperometry and Chronopotentiometry only, so no "
                "plot and no results are shown for this file."
            )

    lsv_samples = [s for s in supported if s["tech"] == "LSV"]
    if lsv_samples:
        st.subheader("Results computed from your files: LSV")
        rows, plots = [], []
        any_unknown = False
        for s in lsv_samples:
            try:
                data, scale = parse_lsv(s["df"], current_unit, area_cm2, rhe_offset)
                if scale == "Unknown (not RHE)":
                    any_unknown = True
                rhe_ok = scale.startswith("RHE")
                tmin = tafel_min if tafel_min > 0 else None
                tmax = tafel_max if tafel_max > 0 else None
                if tmin is not None and tmax is not None and tmin >= tmax:
                    tmin = tmax = None
                m = analyze_lsv(data, "auto", rhe_ok, tmin, tmax)
            except Exception as exc:
                st.error(f"Could not analyse {s['name']}: {exc}")
                continue
            if lsv_summary is None:
                tafel_abs = (abs(m["tafel_slope_mV_dec"])
                             if m["tafel_slope_mV_dec"] is not None else None)
                lsv_summary = {"E_onset_V": m["onset_potential_V"],
                               "tafel": tafel_abs}
            rows.append({
                "Sample": s["name"],
                "Mode": m["mode"],
                "Scale": scale,
                "Onset (V)": fmt(m["onset_potential_V"], 3),
                "E at 10 mA/cm2 (V)": fmt(m["E_at_10_mA_V"], 3),
                "eta10 (mV)": fmt(m["overpotential_10_mV"], 0),
                "E at 100 mA/cm2 (V)": fmt(m["E_at_100_mA_V"], 3),
                "eta100 (mV)": fmt(m["overpotential_100_mV"], 0),
                "Tafel (mV/dec)": fmt(m["tafel_slope_mV_dec"], 0),
                "Tafel R2": fmt(m["tafel_r2"], 3),
                "Tafel points": m["tafel_points"],
                "Max |j| (mA/cm2)": fmt(m["max_abs_current_mA_cm2"], 1),
                "Points": m["points"],
            })
            plots.append((s["name"], data, m, scale))
            candidates.append({
                "label": f"{s['name']} (LSV)",
                "sample": s["name"],
                "tech": "LSV",
                "mode": m["mode"],
                "scale": scale,
                "metrics": {
                    "onset": m["onset_potential_V"],
                    "eta10": m["overpotential_10_mV"],
                    "eta100": m["overpotential_100_mV"],
                    "tafel": m["tafel_slope_mV_dec"],
                    "max_j": m["max_abs_current_mA_cm2"],
                    "rct": None,
                    "retention": None,
                    "drift": None,
                },
            })
            lsv_entries.append({"name": s["name"], "data": data, "m": m,
                                "scale": scale})
        if rows:
            table = pd.DataFrame(rows)
            st.dataframe(table, use_container_width=True)
            st.download_button("Download LSV results as CSV",
                               table.to_csv(index=False).encode("utf-8"),
                               file_name="lsv_results.csv", mime="text/csv")
        if plots:
            first = plots[0][2]
            metric_cards([("Onset (V)", fmt(first["onset_potential_V"], 3)),
                          ("eta10 (mV)", fmt(first["overpotential_10_mV"], 0)),
                          ("Tafel (mV/dec)", fmt(first["tafel_slope_mV_dec"], 0)),
                          ("Max |j| (mA/cm2)", fmt(first["max_abs_current_mA_cm2"], 1))])
        if any_unknown:
            st.warning("Potential reference scale is Unknown for at least one sample, so "
                       "overpotential is not calculated for it. Choose RHE in the sidebar "
                       "when the potentials are already on the RHE scale, or enter a "
                       "verified offset.")
        for idx, (name, data, m, scale) in enumerate(plots):
            st.subheader(f"Curve: {name}")
            curve_key = f"lsv_curve_{idx}"
            marker_key = f"lsv_marker_{idx}"
            curve_color = st.session_state.get(curve_key, "#1f77b4")
            marker_color = st.session_state.get(marker_key, "#ff7f0e")
            fig = lsv_figure(data, m, curve_color, marker_color)
            st.pyplot(fig)
            plt.close(fig)
            picked_curve = st.color_picker("Curve color", value=curve_color, key=curve_key)
            picked_marker = st.color_picker("Marker color", value=marker_color, key=marker_key)
            export_fig = lsv_figure(data, m, picked_curve, picked_marker)
            base = safe_filename(name)
            st.download_button("Download figure as PNG",
                               figure_bytes(export_fig, "png", dpi),
                               file_name=f"{base}_lsv.png", mime="image/png",
                               key=f"lsv_png_{idx}")
            st.download_button("Download figure as TIF",
                               figure_bytes(export_fig, "tiff", dpi),
                               file_name=f"{base}_lsv.tif", mime="image/tiff",
                               key=f"lsv_tif_{idx}")
            plt.close(export_fig)
            if m["overpotential_10_mV"] is not None:
                st.info(f"Overpotential at 10 mA/cm2 is "
                        f"{m['overpotential_10_mV']:.0f} mV. Lower is better, "
                        "less energy wasted as heat.")
            elif scale == "Unknown (not RHE)":
                st.info("Overpotential is not calculated because the potential "
                        "reference scale is Unknown. Choose RHE or a verified "
                        "offset in the sidebar.")
            else:
                st.info("Not enough current range to determine eta10.")
            if m["tafel_slope_mV_dec"] is not None:
                st.info(f"Tafel fit over {fmt(m['tafel_j_min'], 2)} to "
                        f"{fmt(m['tafel_j_max'], 2)} mA/cm2 uses "
                        f"{m['tafel_points']} points, R2 = {fmt(m['tafel_r2'], 3)}.")
                if (m["tafel_r2"] is not None and m["tafel_r2"] < 0.9) or m["tafel_points"] < 10:
                    st.warning("The Tafel fit is weak for this sample. Choose a "
                               "narrower current interval or treat the slope as "
                               "unreliable.")
            else:
                st.info("No Tafel fit available for this sample.")
            with st.expander(f"Tafel Fit Inspector: {name}"):
                st.write("Choose a current interval and inspect the fit: the "
                         "points, the fitted line and the residuals. A good "
                         "fit has small residuals with no clear pattern.")
                insp_min = st.number_input("Inspector fit min (mA/cm2)",
                                           min_value=0.0,
                                           value=float(m["tafel_j_min"]) if m["tafel_j_min"] else 0.0,
                                           step=1.0, key=f"taf_min_{idx}")
                insp_max = st.number_input("Inspector fit max (mA/cm2)",
                                           min_value=0.0,
                                           value=float(m["tafel_j_max"]) if m["tafel_j_max"] else 0.0,
                                           step=1.0, key=f"taf_max_{idx}")
                st.caption("0 and 0 means the automatic band is used.")
                insp = fit_tafel_interval(data["Potential_V"].to_numpy(dtype=float),
                                          data["Current_Density"].to_numpy(dtype=float),
                                          insp_min if insp_min > 0 else None,
                                          insp_max if insp_max > 0 else None)
                if insp is None:
                    st.info("Not enough points in this interval for a fit.")
                else:
                    st.write(f"Inspector fit: slope {fmt(insp['slope_mV_dec'], 0)} mV/dec, "
                             f"R2 = {fmt(insp['r2'], 3)}, points = {insp['points']}, "
                             f"interval {fmt(insp['j_min'], 2)} to {fmt(insp['j_max'], 2)} mA/cm2.")
                    if (insp["r2"] is not None and insp["r2"] < 0.9) or insp["points"] < 10:
                        st.warning("This interval gives a weak fit. Try a narrower interval.")
                    fig = tafel_inspector_figure(insp)
                    st.pyplot(fig)
                    plt.close(fig)
                    base = safe_filename(name)
                    st.download_button("Download inspector figure as PNG",
                                       figure_bytes(tafel_inspector_figure(insp), "png", dpi),
                                       file_name=f"{base}_tafel_inspector.png", mime="image/png",
                                       key=f"taf_png_{idx}")
                    st.download_button("Download inspector figure as TIF",
                                       figure_bytes(tafel_inspector_figure(insp), "tiff", dpi),
                                       file_name=f"{base}_tafel_inspector.tif", mime="image/tiff",
                                       key=f"taf_tif_{idx}")
                    plt.close("all")
        if plots:
            st.subheader("Interpretation: LSV")
            lsv_results = {name: m for name, data, m, scale in plots}
            for line in interpret_lsv(lsv_results):
                st.info(line)

    eis_samples = [s for s in supported if s["tech"] == "EIS"]
    if eis_samples:
        st.subheader("Results computed from your files: EIS")
        rows, plots = [], []
        for s in eis_samples:
            try:
                data = parse_eis(s["df"])
                m = analyze_eis(data)
            except Exception as exc:
                st.error(f"Could not analyse {s['name']}: {exc}")
                continue
            if m["fit_ok"] and eis_summary is None:
                zr = data["Zreal"].to_numpy(dtype=float)
                zi = data["Zimag"].to_numpy(dtype=float)
                fr = data["Freq"].to_numpy(dtype=float)
                peak_freq = (float(fr[int(np.argmax(-zi))])
                             if len(fr) else None)
                phase_min = (float(np.min(np.degrees(np.arctan2(zi, zr))))
                             if len(zr) else None)
                tau_rc = (m["Rct_ohm"] * m["C_F"]
                          if m["Rct_ohm"] and m["C_F"] else None)
                eis_summary = {"Rs_ohm": m["Rs_ohm"],
                               "Rct_ohm": m["Rct_ohm"],
                               "Cdl_est_F": m["C_F"],
                               "peak_freq_Hz": peak_freq,
                               "tau_s": tau_rc,
                               "phase_min": phase_min}
            rows.append({
                "Sample": s["name"],
                "Rs (ohm)": fmt(m["Rs_ohm"], 2),
                "Rct (ohm, RC model)": fmt(m["Rct_ohm"], 2),
                "C (mF, RC model)": fmt(m["C_F"] * 1e3, 3) if m["C_F"] else "Not available",
                "RMSE (ohm)": fmt(m["RMSE_ohm"], 3),
                "Points": m["points"],
            })
            plots.append((s["name"], data, m))
            candidates.append({
                "label": f"{s['name']} (EIS)",
                "sample": s["name"],
                "tech": "EIS",
                "mode": None,
                "scale": "Not applicable",
                "metrics": {
                    "onset": None,
                    "eta10": None,
                    "eta100": None,
                    "tafel": None,
                    "max_j": None,
                    "rct": m["Rct_ohm"] if m["fit_ok"] else None,
                    "retention": None,
                    "drift": None,
                },
            })
            eis_entries.append({"name": s["name"], "m": m})
        if rows:
            table = pd.DataFrame(rows)
            st.dataframe(table, use_container_width=True)
            st.download_button("Download EIS results as CSV",
                               table.to_csv(index=False).encode("utf-8"),
                               file_name="eis_results.csv", mime="text/csv")
        if plots:
            first = plots[0][2]
            metric_cards([("Rs (ohm)", fmt(first["Rs_ohm"], 2)),
                          ("Rct (ohm)", fmt(first["Rct_ohm"], 2)),
                          ("C (mF)", fmt(first["C_F"] * 1e3, 3) if first["C_F"] else "Not available"),
                          ("RMSE (ohm)", fmt(first["RMSE_ohm"], 3))])
        for idx, (name, data, m) in enumerate(plots):
            st.subheader(f"Nyquist plot: {name}")
            points_key = f"eis_points_{idx}"
            fit_key = f"eis_fit_{idx}"
            points_color = st.session_state.get(points_key, "#1f77b4")
            fit_color = st.session_state.get(fit_key, "#d62728")
            fig = eis_figure(data, m, points_color, fit_color)
            st.pyplot(fig)
            plt.close(fig)
            picked_points = st.color_picker("Points color", value=points_color, key=points_key)
            picked_fit = st.color_picker("Fit line color", value=fit_color, key=fit_key)
            export_fig = eis_figure(data, m, picked_points, picked_fit)
            base = safe_filename(name)
            st.download_button("Download figure as PNG",
                               figure_bytes(export_fig, "png", dpi),
                               file_name=f"{base}_eis.png", mime="image/png",
                               key=f"eis_png_{idx}")
            st.download_button("Download figure as TIF",
                               figure_bytes(export_fig, "tiff", dpi),
                               file_name=f"{base}_eis.tif", mime="image/tiff",
                               key=f"eis_tif_{idx}")
            plt.close(export_fig)
            if m["fit_ok"]:
                st.info(f"RC model fit: Rct = {m['Rct_ohm']:.1f} ohm, "
                        f"Rs = {m['Rs_ohm']:.1f} ohm, RMSE = {m['RMSE_ohm']:.3f} ohm. "
                        "Lower Rct suggests faster charge transfer only if this "
                        "simple circuit describes the data.")
                if m["RMSE_ohm"] is not None and m["Rct_ohm"] and m["RMSE_ohm"] > 0.2 * max(m["Rct_ohm"], 1.0):
                    st.warning("The RMSE is large relative to Rct, so the simple "
                               "RC circuit may not describe this spectrum well.")
            else:
                st.info("No RC model fit is available for this spectrum, so no "
                        "Rct is reported.")
            if len(data) and float(np.median(data["Zimag"])) > 0:
                st.info("Z'' is mostly positive in this file. The RC model "
                        "expects signed Z'' (negative for a capacitive arc). "
                        "If your instrument stores minus Z'', the fit may be poor.")
            cpe = None
            try:
                zr = data["Zreal"].to_numpy(dtype=float)
                zi = data["Zimag"].to_numpy(dtype=float)
                fr = data["Freq"].to_numpy(dtype=float)
                rs0 = m["Rs_ohm"] if m["fit_ok"] and m["Rs_ohm"] is not None else float(np.min(zr))
                rct0 = m["Rct_ohm"] if m["fit_ok"] and m["Rct_ohm"] is not None else max(float(np.max(zr) - np.min(zr)), 1.0)
                c0 = m["C_F"] if m["fit_ok"] and m["C_F"] else 1e-3
                cpe = fit_rcpe(fr, zr, zi, rs0, rct0, c0)
            except Exception:
                cpe = None
            if cpe is not None:
                st.write(f"CPE circuit fit: Rs = {fmt(cpe['Rs_ohm'], 2)} ohm, "
                         f"Rct = {fmt(cpe['Rct_ohm'], 2)} ohm, "
                         f"Q = {fmt(cpe['Q'], 3)}, n = {fmt(cpe['n'], 3)}, "
                         f"RMSE = {fmt(cpe['RMSE_ohm'], 3)} ohm.")
                if m["fit_ok"] and m["RMSE_ohm"] is not None:
                    if cpe["RMSE_ohm"] <= m["RMSE_ohm"]:
                        st.info(f"The CPE fit RMSE ({fmt(cpe['RMSE_ohm'], 3)} ohm) is at least as good as the RC fit RMSE ({fmt(m['RMSE_ohm'], 3)} ohm).")
                    else:
                        st.info(f"The CPE fit RMSE ({fmt(cpe['RMSE_ohm'], 3)} ohm) is worse than the RC fit RMSE ({fmt(m['RMSE_ohm'], 3)} ohm), so the RC values are preferred here.")
                if cpe["n"] < 0.8:
                    st.warning(f"The semicircle is strongly depressed (n = {fmt(cpe['n'], 3)}), so a single Rct is indicative only.")
                order = np.argsort(fr)
                neg_zi = -zi
                low_mean = float(np.mean(neg_zi[order][:max(1, len(fr) // 10)]))
                if float(np.max(neg_zi)) > 0 and low_mean > 0.2 * float(np.max(neg_zi)):
                    st.warning("The arc does not close at low frequency, so Rct is uncertain.")
                fig = cpe_nyquist_figure(data, cpe["fit"])
                st.pyplot(fig)
                plt.close(fig)
                base = safe_filename(name)
                st.download_button("Download CPE Nyquist figure as PNG",
                                   figure_bytes(cpe_nyquist_figure(data, cpe["fit"]), "png", dpi),
                                   file_name=f"{base}_cpe_nyquist.png", mime="image/png",
                                   key=f"cpe_nyq_png_{idx}")
                st.download_button("Download CPE Nyquist figure as TIF",
                                   figure_bytes(cpe_nyquist_figure(data, cpe["fit"]), "tiff", dpi),
                                   file_name=f"{base}_cpe_nyquist.tif", mime="image/tiff",
                                   key=f"cpe_nyq_tif_{idx}")
                plt.close("all")
                fig = cpe_bode_figure(fr, zr, zi, cpe["fit"])
                st.pyplot(fig)
                plt.close(fig)
                st.download_button("Download CPE Bode figure as PNG",
                                   figure_bytes(cpe_bode_figure(fr, zr, zi, cpe["fit"]), "png", dpi),
                                   file_name=f"{base}_cpe_bode.png", mime="image/png",
                                   key=f"cpe_bode_png_{idx}")
                st.download_button("Download CPE Bode figure as TIF",
                                   figure_bytes(cpe_bode_figure(fr, zr, zi, cpe["fit"]), "tiff", dpi),
                                   file_name=f"{base}_cpe_bode.tif", mime="image/tiff",
                                   key=f"cpe_bode_tif_{idx}")
                plt.close("all")
                fig = cpe_resid_figure(fr, zr, zi, cpe["fit"])
                st.pyplot(fig)
                plt.close(fig)
                st.download_button("Download CPE residuals figure as PNG",
                                   figure_bytes(cpe_resid_figure(fr, zr, zi, cpe["fit"]), "png", dpi),
                                   file_name=f"{base}_cpe_resid.png", mime="image/png",
                                   key=f"cpe_res_png_{idx}")
                st.download_button("Download CPE residuals figure as TIF",
                                   figure_bytes(cpe_resid_figure(fr, zr, zi, cpe["fit"]), "tiff", dpi),
                                   file_name=f"{base}_cpe_resid.tif", mime="image/tiff",
                                   key=f"cpe_res_tif_{idx}")
                plt.close("all")
            else:
                st.info("No CPE circuit fit is available for this spectrum.")
        if plots:
            st.subheader("Interpretation: EIS")
            eis_results = {name: m for name, data, m in plots}
            for line in interpret_eis(eis_results):
                st.info(line)

    chrono_samples = [s for s in supported if s["tech"] == "Chrono"]
    if chrono_samples:
        st.subheader("Results computed from your files: Chronoamperometry")
        rows, plots = [], []
        for s in chrono_samples:
            try:
                data = parse_chrono(s["df"], current_unit, area_cm2)
                m = analyze_chrono(data)
            except Exception as exc:
                st.error(f"Could not analyse {s['name']}: {exc}")
                continue
            rows.append({
                "Sample": s["name"],
                "Initial (mA/cm2)": fmt(m["initial_current"], 2),
                "Final (mA/cm2)": fmt(m["final_current"], 2),
                "Retention (%)": fmt(m["retention_pct"], 1),
                "Avg (mA/cm2)": fmt(m["avg_current"], 2),
                "Duration (h)": fmt(m["duration"], 3),
                "Points": m["points"],
            })
            plots.append((s["name"], data, m))
            candidates.append({
                "label": f"{s['name']} (Chrono)",
                "sample": s["name"],
                "tech": "Chrono",
                "mode": None,
                "scale": "Not applicable",
                "metrics": {
                    "onset": None,
                    "eta10": None,
                    "eta100": None,
                    "tafel": None,
                    "max_j": None,
                    "rct": None,
                    "retention": m["retention_pct"],
                    "drift": None,
                },
            })
            chrono_entries.append({"name": s["name"], "data": data, "m": m})
        if rows:
            table = pd.DataFrame(rows)
            st.dataframe(table, use_container_width=True)
            st.download_button("Download chrono results as CSV",
                               table.to_csv(index=False).encode("utf-8"),
                               file_name="chrono_results.csv", mime="text/csv")
        if plots:
            first = plots[0][2]
            metric_cards([("Retention (%)", fmt(first["retention_pct"], 1)),
                          ("Avg (mA/cm2)", fmt(first["avg_current"], 2)),
                          ("Duration (h)", fmt(first["duration"], 3))])
        for idx, (name, data, m) in enumerate(plots):
            st.subheader(f"Stability curve: {name}")
            curve_key = f"chrono_curve_{idx}"
            curve_color = st.session_state.get(curve_key, "#1f77b4")
            fig = chrono_figure(data, curve_color)
            st.pyplot(fig)
            plt.close(fig)
            picked_curve = st.color_picker("Curve color", value=curve_color, key=curve_key)
            export_fig = chrono_figure(data, picked_curve)
            base = safe_filename(name)
            st.download_button("Download figure as PNG",
                               figure_bytes(export_fig, "png", dpi),
                               file_name=f"{base}_chrono.png", mime="image/png",
                               key=f"chrono_png_{idx}")
            st.download_button("Download figure as TIF",
                               figure_bytes(export_fig, "tiff", dpi),
                               file_name=f"{base}_chrono.tif", mime="image/tiff",
                               key=f"chrono_tif_{idx}")
            plt.close(export_fig)
            if m["retention_pct"] is not None:
                if m["retention_pct"] > 105:
                    st.info("Current increased during the test. That is surface "
                            "activation, not stability.")
                else:
                    grade = ("excellent" if m["retention_pct"] >= 90 else
                             "good" if m["retention_pct"] >= 80 else
                             "moderate" if m["retention_pct"] >= 65 else "poor")
                    st.info(f"Retention {m['retention_pct']:.1f} percent, "
                            f"{grade} durability.")
        if plots:
            st.subheader("Interpretation: Chronoamperometry")
            chrono_results = {name: m for name, data, m in plots}
            for line in interpret_chrono(chrono_results):
                st.info(line)

    cp_samples = [s for s in supported if s["tech"] == "CP"]
    if cp_samples:
        st.subheader("Results computed from your files: Chronopotentiometry")
        rows, plots = [], []
        for s in cp_samples:
            try:
                data = parse_cp(s["df"], current_unit, area_cm2)
                m = analyze_cp(data)
            except Exception as exc:
                st.error(f"Could not analyse {s['name']}: {exc}")
                continue
            rows.append({
                "Sample": s["name"],
                "Applied current (mA/cm2)": fmt(m["applied_current"], 2),
                "Initial (V)": fmt(m["initial_potential"], 3),
                "Final (V)": fmt(m["final_potential"], 3),
                "Mean (V)": fmt(m["mean_potential"], 3),
                "Drift (mV/h)": fmt(m["drift_mV_h"], 1),
                "Duration (h)": fmt(m["duration"], 3),
                "Points": m["points"],
            })
            plots.append((s["name"], data, m))
            candidates.append({
                "label": f"{s['name']} (CP)",
                "sample": s["name"],
                "tech": "CP",
                "mode": None,
                "scale": "Not applicable",
                "metrics": {
                    "onset": None,
                    "eta10": None,
                    "eta100": None,
                    "tafel": None,
                    "max_j": None,
                    "rct": None,
                    "retention": None,
                    "drift": m["drift_mV_h"],
                },
            })
        if rows:
            table = pd.DataFrame(rows)
            st.dataframe(table, use_container_width=True)
            st.download_button("Download CP results as CSV",
                               table.to_csv(index=False).encode("utf-8"),
                               file_name="cp_results.csv", mime="text/csv")
        if plots:
            first = plots[0][2]
            metric_cards([("Potential drift (mV/h)", fmt(first["drift_mV_h"], 1)),
                          ("Mean potential (V)", fmt(first["mean_potential"], 3)),
                          ("Duration (h)", fmt(first["duration"], 3))])
        for idx, (name, data, m) in enumerate(plots):
            st.subheader(f"Potential curve: {name}")
            curve_key = f"cp_curve_{idx}"
            curve_color = st.session_state.get(curve_key, "#1f77b4")
            fig = cp_figure(data, curve_color)
            st.pyplot(fig)
            plt.close(fig)
            picked_curve = st.color_picker("Curve color", value=curve_color, key=curve_key)
            export_fig = cp_figure(data, picked_curve)
            base = safe_filename(name)
            st.download_button("Download figure as PNG",
                               figure_bytes(export_fig, "png", dpi),
                               file_name=f"{base}_cp.png", mime="image/png",
                               key=f"cp_png_{idx}")
            st.download_button("Download figure as TIF",
                               figure_bytes(export_fig, "tiff", dpi),
                               file_name=f"{base}_cp.tif", mime="image/tiff",
                               key=f"cp_tif_{idx}")
            plt.close(export_fig)
            if m["drift_mV_h"] is not None:
                st.info(f"Potential drift is {m['drift_mV_h']:.1f} mV per hour. "
                        "A smaller absolute drift means a more stable potential "
                        "under constant current.")
        if plots:
            st.subheader("Interpretation: Chronopotentiometry")
            for name, data, m in plots:
                st.info(f"{name}: potential moves from "
                        f"{fmt(m['initial_potential'], 3)} V to "
                        f"{fmt(m['final_potential'], 3)} V over "
                        f"{fmt(m['duration'], 3)} h, drift "
                        f"{fmt(m['drift_mV_h'], 1)} mV per hour.")

    ml_section(eis_summary, lsv_summary)
    pec_section(lsv_entries, eis_entries, chrono_entries)
    comparison_section(candidates)
    st.divider()
    st.markdown('<p style="text-align:center;color:#64748B;font-size:13px;">'
                'ElectroCatalyst AI • Real data only • Every number is '
                'computed from your files</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
