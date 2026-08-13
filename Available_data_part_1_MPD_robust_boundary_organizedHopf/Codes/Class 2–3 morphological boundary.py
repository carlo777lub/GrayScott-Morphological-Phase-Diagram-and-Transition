#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code author: Cruz Castillo Carlos Alberto
Class 2–3 morphological boundary.py

Refined Gray--Scott scan near the lower Hopf branch.
Designed for Leviathan/SLURM array jobs.

Purpose:
  1) Simulate only a band around the lower Hopf curve.
  2) Classify each point as Cte_i, inhibition, or another region.
  3) Reclassify residual label=0 points as temporally variable when
     they satisfy tstd >= TH_TEMPORAL, while preserving label_original.
  4) Extract the Cte_i--inhibition frontier by K columns.
  5) Compare F_frontier(K) against F_Hopf(K).

Visual classes used:
  0 = Ambiguous / unclassified
  1 = Saturation
  2 = Cte_i
  3 = Temporally variable / contact inhibition
  4 = Labyrinths / stationary patterns
  5 = Division / spots
  6 = Senescence
  7 = Cte0 / extinction

Typical SLURM array usage:
  python -u gs_hopf_refined_band_scan.py run \
      --array-from-slurm \
      --n-blocks 30 \
      --outdir out_hopf_band_u01_v09_square \
      --Nx 100 --Ny 100 --T 20000 --dt 1 \
      --init square --u-pert 0.1 --v-pert 0.9 --patch-side 42 \
      --nK 320 --nD 41 \
      --K-min 0.006 --K-max 0.059 \
      --dF-min -0.0008 --dF-max 0.0025

Merge and comparison:
  python -u gs_hopf_refined_band_scan.py merge \
      --outdir out_hopf_band_u01_v09_square \
      --K-clean 0.045 --make-plots
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Lower Hopf curve
# -----------------------------------------------------------------------------
def hopf_low(K: np.ndarray | float) -> np.ndarray | float:
    """Lower branch: F_H(K)=1/2*(sqrt(K)-2K-sqrt(K-4Ksqrt(K)))."""
    Karr = np.asarray(K, dtype=np.float64)
    rad = Karr - 4.0 * Karr * np.sqrt(Karr)
    out = np.full_like(Karr, np.nan, dtype=np.float64)
    m = (Karr >= 0.0) & (rad >= 0.0) & np.isfinite(rad)
    out[m] = 0.5 * (np.sqrt(Karr[m]) - 2.0 * Karr[m] - np.sqrt(rad[m]))
    if np.isscalar(K):
        return float(out)
    return out


# -----------------------------------------------------------------------------
# Parameters and configuration
# -----------------------------------------------------------------------------
@dataclass
class SimParams:
    Nx: int = 100
    Ny: int = 100
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 1.0
    T: int = 20000
    Du: float = 0.16
    Dv: float = 0.08
    a2: float = 1.0
    bc: str = "neumann"  # periodic | neumann
    clip: bool = True
    sample_every: int = 100
    stat_window: int = 3000
    init: str = "square"  # square | center
    u_pert: float = 0.1
    v_pert: float = 0.9
    patch_side: int = 42
    seed: int = 1234
    noise: float = 0.0


@dataclass
class ClassThresholds:
    # Defaults taken from prior .mat files; editable through CLI arguments.
    th_ext: float = 0.02
    th_fix: float = 0.08
    th_sat_mean: float = 0.8
    th_sat_v_abs: float = 0.8
    th_sat_area_frac: float = 0.65
    th_spatial: float = 0.01
    th_temporal: float = 0.001
    th_active_abs: float = 0.08
    th_rel: float = 0.35
    th_peak_count: int = 8
    peak_min_rel: float = 0.45


# -----------------------------------------------------------------------------
# Initialization and Laplacian
# -----------------------------------------------------------------------------
def initial_condition(p: SimParams) -> Tuple[np.ndarray, np.ndarray]:
    U = np.ones((p.Ny, p.Nx), dtype=np.float64)
    V = np.zeros((p.Ny, p.Nx), dtype=np.float64)

    if p.init.lower() == "square":
        side = int(p.patch_side)
        if side <= 0 or side > min(p.Nx, p.Ny):
            raise ValueError(f"invalid patch_side: {side}")
        i0 = (p.Ny - side) // 2
        j0 = (p.Nx - side) // 2
        U[i0:i0 + side, j0:j0 + side] = p.u_pert
        V[i0:i0 + side, j0:j0 + side] = p.v_pert

    elif p.init.lower() == "center":
        i0 = p.Ny // 2
        j0 = p.Nx // 2
        U[i0, j0] = p.u_pert
        V[i0, j0] = p.v_pert

    else:
        raise ValueError("init must be 'square' or 'center'")

    if p.noise > 0:
        rng = np.random.default_rng(p.seed)
        U += p.noise * rng.standard_normal(U.shape)
        V += p.noise * rng.standard_normal(V.shape)
        if p.clip:
            U = np.clip(U, 0.0, 1.0)
            V = np.clip(V, 0.0, 1.0)
    return U, V


def laplacian_periodic(A: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return ((np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) - 2.0 * A) / (dy * dy) +
            (np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1) - 2.0 * A) / (dx * dx))


def laplacian_neumann(A: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # Ghost-reflect: the exterior neighbor equals the boundary value.
    L = np.zeros_like(A)
    L[1:, :] += A[:-1, :]
    L[:-1, :] += A[1:, :]
    L[:, 1:] += A[:, :-1]
    L[:, :-1] += A[:, 1:]
    L[0, :] += A[0, :]
    L[-1, :] += A[-1, :]
    L[:, 0] += A[:, 0]
    L[:, -1] += A[:, -1]
    # dx=dy by default, but keep an approximate general form if dx!=dy.
    if abs(dx - dy) < 1e-15:
        return (L - 4.0 * A) / (dx * dx)
    # Less compact general form for dx != dy.
    Ly = np.zeros_like(A)
    Ly[1:, :] += A[:-1, :]
    Ly[:-1, :] += A[1:, :]
    Ly[0, :] += A[0, :]
    Ly[-1, :] += A[-1, :]
    Ly = (Ly - 2.0 * A) / (dy * dy)

    Lx = np.zeros_like(A)
    Lx[:, 1:] += A[:, :-1]
    Lx[:, :-1] += A[:, 1:]
    Lx[:, 0] += A[:, 0]
    Lx[:, -1] += A[:, -1]
    Lx = (Lx - 2.0 * A) / (dx * dx)
    return Lx + Ly


def laplacian(A: np.ndarray, p: SimParams) -> np.ndarray:
    if p.bc == "periodic":
        return laplacian_periodic(A, p.dx, p.dy)
    if p.bc == "neumann":
        return laplacian_neumann(A, p.dx, p.dy)
    raise ValueError("bc must be periodic or neumann")


# -----------------------------------------------------------------------------
# Metrics and classification
# -----------------------------------------------------------------------------
def count_peaks_simple(V: np.ndarray, thr: float) -> int:
    """Count 8-neighbor local maxima above thr. Lightweight and scipy-free."""
    if V.shape[0] < 3 or V.shape[1] < 3:
        return 0
    C = V[1:-1, 1:-1]
    m = C > thr
    neighs = [
        V[:-2, :-2], V[:-2, 1:-1], V[:-2, 2:],
        V[1:-1, :-2],              V[1:-1, 2:],
        V[2:, :-2],  V[2:, 1:-1],  V[2:, 2:]
    ]
    for N in neighs:
        m &= C >= N
    return int(np.count_nonzero(m))


def simulate_one(F: float, K: float, p: SimParams, th: ClassThresholds) -> Dict[str, float]:
    U, V = initial_condition(p)

    mean_samples: List[float] = []
    start_stat = max(0, int(p.T - p.stat_window))

    for n in range(1, int(p.T) + 1):
        Uold = U
        Vold = V

        Lu = laplacian(Uold, p)
        Lv = laplacian(Vold, p)

        UV2 = p.a2 * Uold * Vold * Vold
        U = Uold + p.dt * (p.Du * Lu - UV2 + F * (1.0 - Uold))
        V = Vold + p.dt * (p.Dv * Lv + UV2 - (F + K) * Vold)

        if p.clip:
            # This reproduces the numerical practice used in previous simulations.
            np.clip(U, 0.0, 1.0, out=U)
            np.clip(V, 0.0, 1.0, out=V)

        if n >= start_stat and (n % p.sample_every == 0):
            mean_samples.append(float(np.mean(V)))

        if not np.isfinite(U).all() or not np.isfinite(V).all():
            return {
                "ok": 0,
                "label": -1,
                "label_original": -1,
                "label_final": -1,
                "reclassification_reason": "invalid_non_finite",
                "meanV": np.nan, "stdV": np.nan,
                "tstd": np.nan, "maxV": np.nan, "area_frac": np.nan,
                "npeaks": np.nan
            }

    meanV = float(np.mean(V))
    stdV = float(np.std(V))
    maxV = float(np.max(V))
    tstd = float(np.std(mean_samples)) if len(mean_samples) > 1 else 0.0

    active_thr = max(th.th_active_abs, th.th_rel * maxV)
    area_frac = float(np.mean(V > active_thr))
    peak_thr = max(th.th_active_abs, th.peak_min_rel * maxV)
    npeaks = count_peaks_simple(V, peak_thr)

    # First pass: keep the original algorithmic label.
    # Second pass: apply an explicit, traceable reclassification of residual
    # label=0 points when their temporal variability exceeds TH_TEMPORAL.
    label_original = classify_point(meanV, stdV, tstd, maxV, area_frac, npeaks, th)
    label_final, reason = reclassify_label0_temporal(label_original, tstd, th)

    return {
        "ok": 1,
        # Keep "label" as the final label for backward compatibility with
        # plotting, frontier extraction, and previous merge scripts.
        "label": int(label_final),
        "label_original": int(label_original),
        "label_final": int(label_final),
        "reclassification_reason": reason,
        "meanV": meanV, "stdV": stdV,
        "tstd": tstd, "maxV": maxV, "area_frac": area_frac,
        "npeaks": int(npeaks)
    }


def classify_point(meanV: float, stdV: float, tstd: float, maxV: float,
                   area_frac: float, npeaks: int, th: ClassThresholds) -> int:
    """
    Minimal classifier focused on the Cte_i--inhibition frontier.

    Adjust th_spatial/th_temporal if exact reproduction of another script is needed.
    In the Hopf band, the key point is to separate:
      Cte_i: nearly homogeneous state with low stdV and low tstd.
      Inhibition: active non-homogeneous/dynamic state with high tstd or stdV.
    """
    if not np.isfinite(meanV + stdV + tstd + maxV + area_frac):
        return -1

    # Extinction / Cte0.
    if maxV < th.th_ext or meanV < 0.5 * th.th_ext:
        return 7

    # Saturation.
    if meanV >= th.th_sat_mean or (maxV >= th.th_sat_v_abs and area_frac >= th.th_sat_area_frac):
        return 1

    # Cte_i: active state that is nearly constant in space and time.
    if (meanV >= th.th_fix) and (stdV < th.th_spatial) and (tstd < th.th_temporal):
        return 2

    # Contact inhibition: active non-homogeneous or temporally variable state.
    # Near Hopf, this is the class that should directly touch Cte_i.
    if (maxV >= th.th_fix) and (area_frac > 0.005) and (tstd >= th.th_temporal):
        return 3

    # Stationary patterns with many peaks: outside the main direct frontier.
    if (maxV >= th.th_fix) and (stdV >= th.th_spatial) and (npeaks >= th.th_peak_count):
        return 4

    # If it is active, non-constant, and not a clear labyrinth, keep it as inhibition.
    if (maxV >= th.th_fix) and (area_frac > 0.005) and (stdV >= th.th_spatial):
        return 3

    return 0


def reclassify_label0_temporal(label: int, tstd: float, th: ClassThresholds) -> Tuple[int, str]:
    """
    Traceable reclassification of residual points.

    Points left as label=0 by the minimal classifier are not replaced
    silently: label_original is preserved and a reason is reported. If the
    residual point has high temporal variability, tstd >= TH_TEMPORAL, it is
    assigned label_final=3, interpreted in this version as temporally
    variable / contact inhibition.
    """
    if int(label) == 0 and np.isfinite(tstd) and float(tstd) >= th.th_temporal:
        return 3, "label0_reclassified_as_temporally_variable_tstd_ge_TH_TEMPORAL"
    return int(label), ""


# -----------------------------------------------------------------------------
# Sweep grid and writers
# -----------------------------------------------------------------------------
def make_K_values(args: argparse.Namespace) -> np.ndarray:
    return np.linspace(args.K_min, args.K_max, args.nK, dtype=np.float64)


def make_dF_values(args: argparse.Namespace) -> np.ndarray:
    return np.linspace(args.dF_min, args.dF_max, args.nD, dtype=np.float64)


def split_indices(n: int, n_blocks: int, block_id: int) -> np.ndarray:
    if block_id < 0 or block_id >= n_blocks:
        raise ValueError(f"block_id={block_id} fuera de rango 0..{n_blocks - 1}")
    return np.array_split(np.arange(n), n_blocks)[block_id]


def get_block_id(args: argparse.Namespace) -> int:
    if args.array_from_slurm:
        val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if val is None:
            raise RuntimeError("--array-from-slurm is active, but SLURM_ARRAY_TASK_ID does not exist")
        return int(val)
    return int(args.block_id)


def write_csv_header_if_needed(path: str, fieldnames: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def run_block(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    block_id = get_block_id(args)

    p = SimParams(
        Nx=args.Nx, Ny=args.Ny, dx=args.dx, dy=args.dy, dt=args.dt, T=args.T,
        Du=args.Du, Dv=args.Dv, a2=args.a2, bc=args.bc, clip=not args.no_clip,
        sample_every=args.sample_every, stat_window=args.stat_window,
        init=args.init, u_pert=args.u_pert, v_pert=args.v_pert,
        patch_side=args.patch_side, seed=args.seed + block_id, noise=args.noise
    )
    th = ClassThresholds(
        th_ext=args.th_ext, th_fix=args.th_fix, th_sat_mean=args.th_sat_mean,
        th_sat_v_abs=args.th_sat_v_abs, th_sat_area_frac=args.th_sat_area_frac,
        th_spatial=args.th_spatial, th_temporal=args.th_temporal,
        th_active_abs=args.th_active_abs, th_rel=args.th_rel,
        th_peak_count=args.th_peak_count, peak_min_rel=args.peak_min_rel
    )

    K_all = make_K_values(args)
    dF_all = make_dF_values(args)
    k_idx_block = split_indices(len(K_all), args.n_blocks, block_id)

    out_csv = os.path.join(args.outdir, f"points_block_{block_id:04d}.csv")
    fieldnames = [
        "block_id", "iK", "idF", "K", "F", "F_hopf", "dF_offset", "ok",
        "label", "label_original", "label_final", "reclassification_reason",
        "meanV", "stdV", "tstd", "maxV", "area_frac", "npeaks", "seconds"
    ]
    write_csv_header_if_needed(out_csv, fieldnames)

    print(f"[block {block_id}] K indices {k_idx_block[0] if len(k_idx_block) else 'none'}.."
          f"{k_idx_block[-1] if len(k_idx_block) else 'none'}  "
          f"nK_block={len(k_idx_block)}  nD={len(dF_all)}", flush=True)
    print(f"[block {block_id}] output: {out_csv}", flush=True)

    total = len(k_idx_block) * len(dF_all)
    done = 0
    t0_block = time.time()

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for iK in k_idx_block:
            K = float(K_all[iK])
            Fh = float(hopf_low(K))
            if not np.isfinite(Fh):
                continue
            for idF, dF in enumerate(dF_all):
                F = Fh + float(dF)
                if F < args.F_min or F > args.F_max:
                    continue

                t0 = time.time()
                res = simulate_one(F, K, p, th)
                seconds = time.time() - t0

                row = {
                    "block_id": block_id, "iK": int(iK), "idF": int(idF),
                    "K": f"{K:.16g}", "F": f"{F:.16g}",
                    "F_hopf": f"{Fh:.16g}", "dF_offset": f"{float(dF):.16g}",
                    "ok": int(res["ok"]),
                    "label": int(res["label"]),
                    "label_original": int(res.get("label_original", res["label"])),
                    "label_final": int(res.get("label_final", res["label"])),
                    "reclassification_reason": res.get("reclassification_reason", ""),
                    "meanV": f"{res['meanV']:.16g}",
                    "stdV": f"{res['stdV']:.16g}",
                    "tstd": f"{res['tstd']:.16g}",
                    "maxV": f"{res['maxV']:.16g}",
                    "area_frac": f"{res['area_frac']:.16g}",
                    "npeaks": int(res["npeaks"]) if np.isfinite(res["npeaks"]) else -1,
                    "seconds": f"{seconds:.6g}",
                }
                writer.writerow(row)
                f.flush()

                done += 1
                if done % max(1, args.print_every) == 0:
                    elapsed = time.time() - t0_block
                    print(f"[block {block_id}] {done}/{total}  K={K:.6g} F={F:.6g} "
                          f"label={row['label']} meanV={row['meanV']} stdV={row['stdV']} "
                          f"tstd={row['tstd']}  elapsed={elapsed/60:.2f} min", flush=True)

    print(f"[block {block_id}] finished in {(time.time() - t0_block)/60:.2f} min", flush=True)


# -----------------------------------------------------------------------------
# Merge and comparison against Hopf
# -----------------------------------------------------------------------------
def load_points_csvs(outdir: str, th_temporal: float = 0.001) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(outdir, "points_block_*.csv")))
    if not files:
        raise FileNotFoundError(f"No encontre points_block_*.csv en {outdir}")

    rows = []
    for path in files:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    label_in = int(row["label"])
                    tstd_value = float(row["tstd"])

                    # New CSVs already contain traceability columns. Old CSVs do not.
                    # For old points_block_*.csv files, apply the same class-0 temporal
                    # reclassification during merge so that a full resimulation is not needed.
                    if "label_original" in row and row.get("label_original", "") != "":
                        label_original = int(row.get("label_original", label_in))
                        label_final = int(row.get("label_final", label_in))
                        reason = row.get("reclassification_reason", "")
                    else:
                        label_original = label_in
                        if label_in == 0 and np.isfinite(tstd_value) and tstd_value >= th_temporal:
                            label_final = 3
                            reason = "legacy_label0_reclassified_as_temporally_variable_tstd_ge_TH_TEMPORAL"
                        else:
                            label_final = label_in
                            reason = ""

                    # Keep label as the final label used by plots/frontier extraction.
                    label = label_final

                    rows.append((
                        int(row["block_id"]), int(row["iK"]), int(row["idF"]),
                        float(row["K"]), float(row["F"]), float(row["F_hopf"]),
                        float(row["dF_offset"]), int(row["ok"]), label,
                        label_original, label_final, reason,
                        float(row["meanV"]), float(row["stdV"]), tstd_value,
                        float(row["maxV"]), float(row["area_frac"]), int(row["npeaks"]),
                        float(row["seconds"])
                    ))
                except Exception:
                    pass

    dtype = [
        ("block_id", "i4"), ("iK", "i4"), ("idF", "i4"),
        ("K", "f8"), ("F", "f8"), ("F_hopf", "f8"), ("dF_offset", "f8"),
        ("ok", "i4"), ("label", "i4"),
        ("label_original", "i4"), ("label_final", "i4"),
        ("reclassification_reason", "U96"),
        ("meanV", "f8"), ("stdV", "f8"),
        ("tstd", "f8"), ("maxV", "f8"), ("area_frac", "f8"),
        ("npeaks", "i4"), ("seconds", "f8")
    ]
    return np.array(rows, dtype=dtype)


def extract_frontier(points: np.ndarray) -> np.ndarray:
    """
    Extract the Cte_i--inhibition frontier for each iK column.
    Search for neighboring F values with labels 2 and 3. If several crossings exist, use the one closest to Hopf.
    """
    out = []
    for iK in np.unique(points["iK"]):
        sub = points[(points["iK"] == iK) & (points["ok"] == 1)]
        if len(sub) < 2:
            continue
        sub = np.sort(sub, order="F")
        candidates = []
        for a, b in zip(sub[:-1], sub[1:]):
            la, lb = int(a["label"]), int(b["label"])
            if {la, lb} == {2, 3}:
                K = 0.5 * (float(a["K"]) + float(b["K"]))
                F_front = 0.5 * (float(a["F"]) + float(b["F"]))
                Fh = float(hopf_low(K))
                candidates.append((abs(F_front - Fh), K, F_front, Fh, la, lb,
                                   float(a["F"]), float(b["F"])))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, K, F_front, Fh, la, lb, Fa, Fb = candidates[0]
        out.append((int(iK), K, F_front, Fh, F_front - Fh,
                    (F_front - Fh), la, lb, Fa, Fb))

    dtype = [
        ("iK", "i4"), ("K", "f8"), ("F_front", "f8"), ("F_hopf", "f8"),
        ("DeltaF", "f8"), ("DeltaF_raw", "f8"), ("label_a", "i4"), ("label_b", "i4"),
        ("F_a", "f8"), ("F_b", "f8")
    ]
    return np.array(out, dtype=dtype)


def write_structured_csv(path: str, arr: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(arr.dtype.names)
        for row in arr:
            w.writerow([row[name] for name in arr.dtype.names])


def stats_from_frontier(front: np.ndarray, dF_grid: float, K_clean: float | None = None) -> Dict[str, float]:
    if K_clean is not None:
        data = front[front["K"] <= K_clean]
    else:
        data = front
    if len(data) == 0:
        return {}
    d = data["DeltaF"]
    ad = np.abs(d)
    return {
        "N": int(len(data)),
        "K_min": float(np.min(data["K"])),
        "K_max": float(np.max(data["K"])),
        "mean_DeltaF": float(np.mean(d)),
        "median_DeltaF": float(np.median(d)),
        "rmse_DeltaF": float(np.sqrt(np.mean(d * d))),
        "max_abs_DeltaF": float(np.max(ad)),
        "mean_DeltaF_over_dFgrid": float(np.mean(d / dF_grid)),
        "median_DeltaF_over_dFgrid": float(np.median(d / dF_grid)),
        "max_abs_DeltaF_over_dFgrid": float(np.max(ad / dF_grid)),
        "pct_within_1_dFgrid": float(100.0 * np.mean(ad <= 1.0 * dF_grid)),
        "pct_within_1p5_dFgrid": float(100.0 * np.mean(ad <= 1.5 * dF_grid)),
        "pct_within_2_dFgrid": float(100.0 * np.mean(ad <= 2.0 * dF_grid)),
        "pct_positive_DeltaF": float(100.0 * np.mean(d > 0.0)),
    }


def write_stats_csv(path: str, stats: Dict[str, float]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, v])


def write_reclassification_audit_csv(path: str, points: np.ndarray) -> None:
    """Write only the points changed by the label-0 temporal reclassification."""
    fields = [
        "block_id", "iK", "idF", "K", "F", "F_hopf", "dF_offset",
        "label_original", "label_final", "reclassification_reason",
        "meanV", "stdV", "tstd", "maxV", "area_frac", "npeaks"
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for row in points:
            w.writerow([row[name] for name in fields])


def make_plots(outdir: str, points: np.ndarray, front: np.ndarray, dF_grid: float, K_clean: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Band map colored by class.
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ok = points["ok"] == 1
    sc = ax.scatter(points["K"][ok], points["F"][ok], c=points["label"][ok], s=6, cmap="tab10", vmin=0, vmax=7)
    if len(front):
        ax.plot(front["K"], front["F_front"], "k.-", lw=1.5, ms=3, label="extracted frontier")
    Kline = np.linspace(np.nanmin(points["K"]), min(np.nanmax(points["K"]), 1.0/16.0), 1000)
    ax.plot(Kline, hopf_low(Kline), color="magenta", lw=2.0, label="Hopf baja")
    ax.set_xlabel("K")
    ax.set_ylabel("F")
    ax.set_title("Refined scan near Hopf")
    ax.legend(loc="best")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("label")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_band_labels_frontier_hopf.png"))
    plt.close(fig)

    # Normalized residual.
    if len(front):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.plot(front["K"], front["DeltaF"] / dF_grid, "ko-", lw=1.0, ms=3)
        ax.axhline(0, color="magenta", lw=2)
        ax.axhline(1, color="k", lw=1, ls="--")
        ax.axhline(-1, color="k", lw=1, ls="--")
        ax.axhline(2, color="k", lw=1, ls=":")
        ax.axhline(-2, color="k", lw=1, ls=":")
        ax.axvline(K_clean, color="red", lw=1, ls="--")
        ax.set_xlabel("K")
        ax.set_ylabel(r"$\Delta F / \Delta F_{grid}$")
        ax.set_title("Normalized residual: Cte_i-inhibition frontier vs Hopf")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_residuo_normalizado.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.hist(front["DeltaF"] / dF_grid, bins=35, edgecolor="k", alpha=0.7)
        ax.axvline(0, color="magenta", lw=2)
        for x, ls in [(-2, ":"), (-1, "--"), (1, "--"), (2, ":")]:
            ax.axvline(x, color="k", lw=1, ls=ls)
        ax.set_xlabel(r"$\Delta F / \Delta F_{grid}$")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of the normalized residual")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_hist_residuo_normalizado.png"))
        plt.close(fig)


def merge_results(args: argparse.Namespace) -> None:
    points = load_points_csvs(args.outdir, th_temporal=args.th_temporal)
    if len(points) == 0:
        raise RuntimeError("No valid points available to merge")

    points_all_csv = os.path.join(args.outdir, "all_points_merged.csv")
    write_structured_csv(points_all_csv, points)

    # Small audit table for traceability of the class-0 reclassification.
    audit_csv = os.path.join(args.outdir, "reclassification_audit_label0_temporal.csv")
    mask_reclassified = (points["label_original"] == 0) & (points["label_final"] == 3)
    write_reclassification_audit_csv(audit_csv, points[mask_reclassified])

    front = extract_frontier(points)
    if len(front) == 0:
        print("WARNING: direct label 2--3 frontier was not found.")
    front_csv = os.path.join(args.outdir, "frontera_Ctei_Inhibition_vs_Hopf_refinada.csv")
    write_structured_csv(front_csv, front)

    # Transverse sweep resolution: step between dF offsets.
    if args.nD > 1:
        dF_grid = abs((args.dF_max - args.dF_min) / (args.nD - 1))
    else:
        # If sweep arguments are not passed to merge, estimate it from the points.
        dvals = np.unique(np.round(points["dF_offset"], 15))
        dF_grid = float(np.median(np.diff(np.sort(dvals)))) if len(dvals) > 1 else np.nan

    stats_global = stats_from_frontier(front, dF_grid, None)
    stats_global["dF_grid_offset"] = dF_grid
    write_stats_csv(os.path.join(args.outdir, "estadisticas_frontera_vs_Hopf_global.csv"), stats_global)

    if args.K_clean is not None:
        stats_clean = stats_from_frontier(front, dF_grid, args.K_clean)
        stats_clean["K_clean"] = args.K_clean
        stats_clean["dF_grid_offset"] = dF_grid
        write_stats_csv(os.path.join(args.outdir, "estadisticas_frontera_vs_Hopf_zona_limpia.csv"), stats_clean)

    print("\nMerge completed")
    print(f"  merged points: {len(points)}")
    print(f"  extracted frontier: {len(front)}")
    print(f"  label=0 points reclassified as label=3: {int(np.count_nonzero(mask_reclassified))}")
    print(f"  TH_TEMPORAL used during merge: {args.th_temporal}")
    print(f"  points CSV: {points_all_csv}")
    print(f"  reclassification audit CSV: {audit_csv}")
    print(f"  frontier CSV: {front_csv}")
    print("\nGlobal statistics:")
    for k, v in stats_global.items():
        print(f"  {k}: {v}")

    if args.make_plots:
        make_plots(args.outdir, points, front, dF_grid, args.K_clean if args.K_clean is not None else 0.045)
        print("\nFigures saved in outdir.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", type=str, default="out_hopf_band_refined")
    parser.add_argument("--nK", type=int, default=701)
    parser.add_argument("--nD", type=int, default=101)
    parser.add_argument("--K-min", dest="K_min", type=float, default=0.006)
    parser.add_argument("--K-max", dest="K_max", type=float, default=0.059)
    parser.add_argument("--F-min", dest="F_min", type=float, default=0.0)
    parser.add_argument("--F-max", dest="F_max", type=float, default=0.1)
    parser.add_argument("--dF-min", dest="dF_min", type=float, default=-0.0008)
    parser.add_argument("--dF-max", dest="dF_max", type=float, default=0.0025)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refined Gray-Scott scan near the lower Hopf branch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run one sweep block")
    add_common_args(run)
    run.add_argument("--array-from-slurm", action="store_true")
    run.add_argument("--block-id", type=int, default=0)
    run.add_argument("--n-blocks", type=int, default=30)
    run.add_argument("--print-every", type=int, default=1)

    run.add_argument("--Nx", type=int, default=100)
    run.add_argument("--Ny", type=int, default=100)
    run.add_argument("--dx", type=float, default=1.0)
    run.add_argument("--dy", type=float, default=1.0)
    run.add_argument("--dt", type=float, default=1.0)
    run.add_argument("--T", type=int, default=20000)
    run.add_argument("--Du", type=float, default=0.16)
    run.add_argument("--Dv", type=float, default=0.08)
    run.add_argument("--a2", type=float, default=1.0)
    run.add_argument("--bc", choices=["periodic", "neumann"], default="neumann")
    run.add_argument("--no-clip", action="store_true")
    run.add_argument("--sample-every", type=int, default=100)
    run.add_argument("--stat-window", type=int, default=3000)
    run.add_argument("--init", choices=["square", "center"], default="square")
    run.add_argument("--u-pert", type=float, default=0.50)
    run.add_argument("--v-pert", type=float, default=0.25)
    run.add_argument("--patch-side", type=int, default=42)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--noise", type=float, default=0.0)

    # Classification thresholds. Defaults read from prior .mat files.
    run.add_argument("--th-ext", type=float, default=0.02)
    run.add_argument("--th-fix", type=float, default=0.08)
    run.add_argument("--th-sat-mean", type=float, default=0.8)
    run.add_argument("--th-sat-v-abs", type=float, default=0.8)
    run.add_argument("--th-sat-area-frac", type=float, default=0.65)
    run.add_argument("--th-spatial", type=float, default=0.01)
    run.add_argument("--th-temporal", type=float, default=0.001)
    run.add_argument("--th-active-abs", type=float, default=0.08)
    run.add_argument("--th-rel", type=float, default=0.35)
    run.add_argument("--th-peak-count", type=int, default=8)
    run.add_argument("--peak-min-rel", type=float, default=0.45)

    merge = sub.add_parser("merge", help="Merge blocks and extract the frontier")
    add_common_args(merge)
    merge.add_argument("--K-clean", type=float, default=0.045)
    merge.add_argument("--th-temporal", type=float, default=0.001,
                       help="Threshold used during merge to reclassify label=0 as temporally variable")
    merge.add_argument("--make-plots", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "run":
        run_block(args)
    elif args.cmd == "merge":
        merge_results(args)
    else:
        raise RuntimeError("unknown cmd")


if __name__ == "__main__":
    main()
