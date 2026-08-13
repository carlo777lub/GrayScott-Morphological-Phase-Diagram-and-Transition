#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code author: Cruz Castillo Carlos Alberto
gray_scott_hopf_diffusive_modes_deltaF.py

Revised version for high-performance computing.

Single objective:
  For each mode q_{mn}, compute the vertical difference

      DeltaF = F_Hopf(K) - F_intersection(K)

  where F_intersection(K) is extracted from the zero contours:
      Re(lambda_+) = 0,
      Im(lambda_+) = 0,
      Re(lambda_-) = 0,
      Im(lambda_-) = 0,
  for each homogeneous branch:
      v0+ and v0-.

Output:
  <prefix>_Hopf_minus_intersection_only.mat
  <prefix>_diff_summary.csv
  <prefix>_contour_diagnostics.csv
  <prefix>_diff_points.csv  only if --save-point-table is enabled

Conventions:
  x-axis = K
  y-axis = F
  branch_id = +1 -> v0+
  branch_id = -1 -> v0-
  root_id   = +1 -> lambda+
  root_id   = -1 -> lambda-
  kind_id   = 1  -> Re(lambda)=0
  kind_id   = 2  -> Im(lambda)=0
  hopf_branch_id = -1 -> lower Hopf branch
  hopf_branch_id =  0 -> nearest Hopf branch point by point
  hopf_branch_id = +1 -> upper Hopf branch

Revision notes:
  * Does not save full lambda fields or large auxiliary maps.
  * Includes diagnostics to detect flat or poorly defined contours.
  * Allows filtering to the real physical region with --physical-only.
  * Avoids argparse.BooleanOptionalAction for compatibility with older Python versions.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import savemat


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Add --name / --no-name pairs compatible with Python >=3.7."""
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text + " Default: on." if default else help_text + " Default: off.")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help="Disable: " + help_text)
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute only DeltaF = F_Hopf - F_intersection for Gray-Scott modes."
    )

    # Parameter domain: K horizontal, F vertical.
    p.add_argument("--Fmin", type=float, default=0.0)
    p.add_argument("--Fmax", type=float, default=0.1)
    p.add_argument("--Kmin", type=float, default=0.0)
    p.add_argument("--Kmax", type=float, default=0.08)
    p.add_argument("--nF", type=int, default=1000)
    p.add_argument("--nK", type=int, default=1000)

    # Diffusion and numerical domain.
    p.add_argument("--Du", type=float, default=0.16)
    p.add_argument("--Dv", type=float, default=0.08)
    p.add_argument("--Nx", type=int, default=200)
    p.add_argument("--Ny", type=int, default=200)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--dy", type=float, default=1.0)
    p.add_argument(
        "--q-type",
        choices=["discrete", "continuous"],
        default="discrete",
        help="discrete uses the 5-point Laplacian; continuous uses pi/L.",
    )

    # Modes. By default, include q=0.
    add_bool_arg(p, "include-q0", True, "Include q=0 as the first mode.")
    p.add_argument("--n-modes", type=int, default=10)
    p.add_argument("--mode-start", type=int, default=0)
    p.add_argument("--mode-stop", type=int, default=None)

    # Contours and comparison with Hopf.
    p.add_argument(
        "--diff-kinds",
        choices=["re", "im", "both"],
        default="both",
        help="Intersections to compare with Hopf: Re(lambda)=0, Im(lambda)=0, or both.",
    )
    p.add_argument(
        "--hopf-branch",
        choices=["lower", "upper", "nearest", "both"],
        default="lower",
        help=(
            "Hopf branch used for F_Hopf. lower = lower branch; "
            "upper = upper branch; nearest = nearest branch to the point; both = save both."
        ),
    )
    p.add_argument("--contour-min-points", type=int, default=8)
    p.add_argument(
        "--zero-range-tol",
        type=float,
        default=1e-12,
        help="If max(Z)-min(Z) is smaller than this threshold, no contour is extracted.",
    )
    p.add_argument(
        "--zero-level-tol",
        type=float,
        default=1e-10,
        help="Tolerance used to diagnose the fraction of points close to Z=0.",
    )
    add_bool_arg(
        p,
        "physical-only",
        False,
        "Filter intersection points to the nontrivial real region: discriminant>=0, F>eps and F+K>eps.",
    )
    p.add_argument(
        "--eps-nontrivial",
        type=float,
        default=1e-14,
        help="Threshold used by --physical-only to exclude F=0 and F+K=0.",
    )
    add_bool_arg(
        p,
        "save-point-table",
        True,
        "Save a point-by-point table [K,F_intersection,F_Hopf,DeltaF].",
    )
    add_bool_arg(p, "csv", True, "Also save summary and diagnostic CSV files.")

    # Output.
    p.add_argument("--outdir", type=str, default="out_hopf_minus_intersection")
    p.add_argument("--prefix", type=str, default="GS_lambda_q")
    p.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Stored Kv/Fv type. Internal calculations use complex128/float64.",
    )

    return p.parse_args()


def make_modes(
    n_modes: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    q_type: str = "discrete",
    include_q0: bool = True,
) -> np.ndarray:
    """Generate the first n_modes pairs (m,n), sorted by increasing q."""
    box = max(8, int(math.ceil(math.sqrt(max(n_modes, 1)))) + 25)

    while True:
        rows: List[Tuple[float, float, int, int]] = []
        for m in range(box + 1):
            for n in range(box + 1):
                if (m == 0 and n == 0) and not include_q0:
                    continue

                if q_type == "discrete":
                    q2 = (
                        4.0 / dx**2 * math.sin(m * math.pi / (2.0 * (nx - 1))) ** 2
                        + 4.0 / dy**2 * math.sin(n * math.pi / (2.0 * (ny - 1))) ** 2
                    )
                else:
                    lx = (nx - 1) * dx
                    ly = (ny - 1) * dy
                    q2 = (m * math.pi / lx) ** 2 + (n * math.pi / ly) ** 2

                rows.append((q2, math.sqrt(q2), m, n))

        rows.sort(key=lambda r: (r[0], r[2], r[3]))
        if len(rows) >= n_modes:
            rows = rows[:n_modes]
            break
        box *= 2

    out = np.zeros((len(rows), 4), dtype=np.float64)
    for i, (q2, q, m, n) in enumerate(rows):
        out[i, 0] = m
        out[i, 1] = n
        out[i, 2] = q
        out[i, 3] = q2
    return out


def v0_branches(F: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute v0+ and v0- as a complex extension."""
    disc = F**2 - 4.0 * F * (F + K) ** 2
    denom = 2.0 * (F + K)
    sqrt_disc = np.sqrt(disc.astype(np.complex128))

    with np.errstate(divide="ignore", invalid="ignore"):
        v0p = (F + sqrt_disc) / denom
        v0m = (F - sqrt_disc) / denom

    bad = ~np.isfinite(denom) | (denom == 0.0)
    if np.any(bad):
        v0p = v0p.copy()
        v0m = v0m.copy()
        v0p[bad] = np.nan + 1j * np.nan
        v0m[bad] = np.nan + 1j * np.nan

    return v0p, v0m


def lambda_pair(
    F: np.ndarray,
    K: np.ndarray,
    v0: np.ndarray,
    q2: float,
    Du: float,
    Dv: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lambda+ and lambda- for one v0 branch and a given q^2."""
    v02 = v0**2
    T = K - v02 - (Du + Dv) * q2
    Delta = (
        (F + K) * (v02 - F)
        + (Dv * (v02 + F) - Du * (F + K)) * q2
        + Du * Dv * q2**2
    )
    rad = T**2 - 4.0 * Delta
    rt = np.sqrt(rad.astype(np.complex128))
    lam_plus = (T + rt) / 2.0
    lam_minus = (T - rt) / 2.0
    return lam_plus, lam_minus


def contour_segments(
    Kv: np.ndarray,
    Fv: np.ndarray,
    Z: np.ndarray,
    level: float = 0.0,
    min_points: int = 8,
    zero_range_tol: float = 1e-12,
) -> Tuple[List[np.ndarray], Tuple[float, float, float, int]]:
    """
    Extract contours as a list of [K,F] arrays.
    Also return diagnostics: zmin, zmax, zspan, skipped_flag.
    skipped_flag: 0=ok, 1=no finite values, 2=level not crossed,
                  3=almost-flat field.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Zr = np.asarray(Z, dtype=float)
    finite = np.isfinite(Zr)
    if not np.any(finite):
        return [], (np.nan, np.nan, np.nan, 1)

    zmin = float(np.nanmin(Zr))
    zmax = float(np.nanmax(Zr))
    zspan = zmax - zmin

    if zspan <= zero_range_tol:
        return [], (zmin, zmax, zspan, 3)
    if not (zmin - zero_range_tol <= level <= zmax + zero_range_tol):
        return [], (zmin, zmax, zspan, 2)

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = ax.contour(Kv, Fv, Zr, levels=[level])
        segs = []
        if len(cs.allsegs) > 0:
            for seg in cs.allsegs[0]:
                if seg.shape[0] >= min_points:
                    segs.append(np.asarray(seg, dtype=np.float64))
    finally:
        plt.close(fig)
    return segs, (zmin, zmax, zspan, 0)


def hopf_values(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the lower and upper Hopf branches: F_H^-(K), F_H^+(K)."""
    K = np.asarray(K, dtype=np.float64)
    rad = K - 4.0 * K * np.sqrt(K)

    # Rounding correction near K=1/16.
    rad = np.where((rad < 0.0) & (np.abs(rad) < 1e-14), 0.0, rad)
    valid = (K >= 0.0) & (K <= 1.0 / 16.0) & (rad >= 0.0)

    sqrt_rad = np.sqrt(np.where(valid, rad, np.nan))
    base = np.sqrt(np.where(K >= 0.0, K, np.nan)) - 2.0 * K

    F_lower = 0.5 * (base - sqrt_rad)
    F_upper = 0.5 * (base + sqrt_rad)

    F_lower = np.where(valid, F_lower, np.nan)
    F_upper = np.where(valid, F_upper, np.nan)
    return F_lower, F_upper


def physical_mask_points(Kc: np.ndarray, Fc: np.ndarray, eps: float) -> np.ndarray:
    """Mask for the real nontrivial Gray-Scott equilibrium."""
    disc = Fc**2 - 4.0 * Fc * (Fc + Kc) ** 2
    return (
        np.isfinite(Kc)
        & np.isfinite(Fc)
        & (disc >= -1e-12)
        & (Fc > eps)
        & ((Fc + Kc) > eps)
    )


def compute_diff_for_segment(
    seg: np.ndarray,
    hopf_branch: str,
    physical_only: bool,
    eps_nontrivial: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Receive a [K,F_intersection] segment.
    Return K, F_intersection, F_Hopf, DeltaF=F_Hopf-F_intersection.
    For hopf_branch='both', call this function separately with lower and upper.
    """
    Kc = seg[:, 0]
    Fc = seg[:, 1]
    F_lower, F_upper = hopf_values(Kc)

    if hopf_branch == "lower":
        Fh = F_lower
    elif hopf_branch == "upper":
        Fh = F_upper
    elif hopf_branch == "nearest":
        dlow = np.abs(F_lower - Fc)
        dup = np.abs(F_upper - Fc)
        Fh = np.where(dlow <= dup, F_lower, F_upper)
    else:
        raise ValueError("hopf_branch must be lower, upper, or nearest in this function.")

    valid = np.isfinite(Kc) & np.isfinite(Fc) & np.isfinite(Fh)
    if physical_only:
        valid &= physical_mask_points(Kc, Fc, eps_nontrivial)

    Kc = Kc[valid]
    Fc = Fc[valid]
    Fh = Fh[valid]
    dF = Fh - Fc
    return Kc, Fc, Fh, dF


def summarize_diff(
    Kc: np.ndarray,
    Fc: np.ndarray,
    Fh: np.ndarray,
    dF: np.ndarray,
) -> Tuple[float, ...]:
    """DeltaF statistics. Return NaN if there are no points."""
    if Kc.size == 0:
        return tuple([np.nan] * 16)

    abs_dF = np.abs(dF)
    imax = int(np.nanargmax(abs_dF))
    return (
        float(Kc.size),
        float(np.nanmin(Kc)),
        float(np.nanmax(Kc)),
        float(np.nanmin(Fc)),
        float(np.nanmax(Fc)),
        float(np.nanmean(dF)),
        float(np.nanstd(dF)),
        float(np.nanmean(abs_dF)),
        float(np.nanmedian(abs_dF)),
        float(np.nanpercentile(abs_dF, 95.0)),
        float(np.sqrt(np.nanmean(dF**2))),
        float(np.nanmax(abs_dF)),
        float(Kc[imax]),
        float(Fc[imax]),
        float(Fh[imax]),
        float(dF[imax]),
    )


def write_csv(path: Path, data: np.ndarray, columns: List[str]) -> None:
    """Save a numeric CSV table with headers."""
    header = ",".join(columns)
    if data.size == 0:
        path.write_text(header + "\n", encoding="utf-8")
        return
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.17g")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fdtype = np.float32 if args.dtype == "float32" else np.float64

    Kv = np.linspace(args.Kmin, args.Kmax, args.nK, dtype=np.float64)
    Fv = np.linspace(args.Fmin, args.Fmax, args.nF, dtype=np.float64)
    K, F = np.meshgrid(Kv, Fv)

    v0p, v0m = v0_branches(F, K)

    modes_all = make_modes(
        args.n_modes, args.Nx, args.Ny, args.dx, args.dy, args.q_type, args.include_q0
    )
    mode_stop = args.n_modes if args.mode_stop is None else args.mode_stop
    mode_stop = min(mode_stop, modes_all.shape[0])
    mode_start = max(0, args.mode_start)
    modes = modes_all[mode_start:mode_stop]

    if args.diff_kinds == "re":
        kind_specs = [(1, "Re")]
    elif args.diff_kinds == "im":
        kind_specs = [(2, "Im")]
    else:
        kind_specs = [(1, "Re"), (2, "Im")]

    if args.hopf_branch == "both":
        hopf_specs = [(-1, "lower"), (+1, "upper")]
    else:
        hb_id = {"lower": -1, "upper": +1, "nearest": 0}[args.hopf_branch]
        hopf_specs = [(hb_id, args.hopf_branch)]

    combo_specs = [(+1, "v0p", v0p), (-1, "v0m", v0m)]
    root_specs = [(+1, "lambda_plus"), (-1, "lambda_minus")]

    diff_summary_rows: List[List[float]] = []
    diff_point_rows: List[np.ndarray] = []
    diagnostic_rows: List[List[float]] = []

    print("=== Gray-Scott: only F_Hopf - F_intersection v2 ===")
    print(f"Grid: nF={args.nF}, nK={args.nK}, total={args.nF * args.nK:,}")
    print(f"F range: [{args.Fmin}, {args.Fmax}], K range: [{args.Kmin}, {args.Kmax}]")
    print(f"Modes requested: {args.n_modes}; computing [{mode_start}, {mode_stop})")
    print(f"include_q0={args.include_q0}; hopf_branch={args.hopf_branch}; diff_kinds={args.diff_kinds}")
    print(f"physical_only={args.physical_only}; save_point_table={args.save_point_table}")
    print(f"Saving to: {outdir.resolve()}")

    for local_i, row in enumerate(modes):
        global_mode_idx = mode_start + local_i
        m = int(row[0])
        n = int(row[1])
        q = float(row[2])
        q2 = float(row[3])
        print(f"Mode {global_mode_idx:04d}: (m,n)=({m},{n}), q={q:.10g}", flush=True)

        for branch_id, branch_name, v0 in combo_specs:
            lam_plus, lam_minus = lambda_pair(F, K, v0, q2, args.Du, args.Dv)
            lam_by_root = {+1: lam_plus, -1: lam_minus}

            for root_id, root_name in root_specs:
                lam = lam_by_root[root_id]

                for kind_id, kind_name in kind_specs:
                    Z = np.real(lam) if kind_id == 1 else np.imag(lam)
                    finite = np.isfinite(Z)
                    zero_fraction = float(np.mean(np.abs(Z[finite]) <= args.zero_level_tol)) if np.any(finite) else np.nan

                    segs, diag = contour_segments(
                        Kv,
                        Fv,
                        Z,
                        level=0.0,
                        min_points=args.contour_min_points,
                        zero_range_tol=args.zero_range_tol,
                    )
                    zmin, zmax, zspan, skipped_flag = diag
                    diagnostic_rows.append(
                        [
                            float(global_mode_idx), float(m), float(n), q, q2,
                            float(branch_id), float(root_id), float(kind_id),
                            float(len(segs)), zmin, zmax, zspan,
                            zero_fraction, float(skipped_flag),
                        ]
                    )

                    for curve_id, seg in enumerate(segs, start=1):
                        for hopf_branch_id, hopf_branch_name in hopf_specs:
                            Kc, Fc, Fh, dF = compute_diff_for_segment(
                                seg,
                                hopf_branch_name,
                                physical_only=args.physical_only,
                                eps_nontrivial=args.eps_nontrivial,
                            )

                            stats = summarize_diff(Kc, Fc, Fh, dF)
                            if np.isnan(stats[0]):
                                continue

                            diff_summary_rows.append(
                                [
                                    float(global_mode_idx),
                                    float(m),
                                    float(n),
                                    q,
                                    q2,
                                    float(branch_id),
                                    float(root_id),
                                    float(kind_id),
                                    float(curve_id),
                                    float(hopf_branch_id),
                                    *stats,
                                ]
                            )

                            if args.save_point_table and Kc.size > 0:
                                # [mode,m,n,q,q2,branch,root,kind,curve,hopf_branch,K,F_intersection,F_Hopf,DeltaF,abs_DeltaF]
                                meta = np.column_stack(
                                    [
                                        np.full(Kc.size, float(global_mode_idx)),
                                        np.full(Kc.size, float(m)),
                                        np.full(Kc.size, float(n)),
                                        np.full(Kc.size, q),
                                        np.full(Kc.size, q2),
                                        np.full(Kc.size, float(branch_id)),
                                        np.full(Kc.size, float(root_id)),
                                        np.full(Kc.size, float(kind_id)),
                                        np.full(Kc.size, float(curve_id)),
                                        np.full(Kc.size, float(hopf_branch_id)),
                                        Kc,
                                        Fc,
                                        Fh,
                                        dF,
                                        np.abs(dF),
                                    ]
                                )
                                diff_point_rows.append(meta)

    diff_summary_table = (
        np.asarray(diff_summary_rows, dtype=np.float64)
        if diff_summary_rows
        else np.empty((0, 26), dtype=np.float64)
    )
    diff_point_table = (
        np.vstack(diff_point_rows).astype(np.float64)
        if diff_point_rows
        else np.empty((0, 15), dtype=np.float64)
    )
    diagnostic_table = (
        np.asarray(diagnostic_rows, dtype=np.float64)
        if diagnostic_rows
        else np.empty((0, 14), dtype=np.float64)
    )

    # Hopf table over the domain for reference.
    Kh = np.linspace(max(0.0, args.Kmin), min(args.Kmax, 1.0 / 16.0), 3000)
    Fh_lo, Fh_up = hopf_values(Kh)
    hopf_table = np.column_stack([Kh, Fh_lo, Fh_up])

    modes_computed = modes.astype(np.float64)
    params_json = json.dumps(vars(args).copy(), indent=2)

    summary_columns = [
        "mode_index",
        "m",
        "n",
        "q",
        "q2",
        "branch_id",
        "root_id",
        "kind_id",
        "curve_id",
        "hopf_branch_id",
        "n_points",
        "K_min",
        "K_max",
        "F_intersection_min",
        "F_intersection_max",
        "mean_DeltaF",
        "std_DeltaF",
        "mean_abs_DeltaF",
        "median_abs_DeltaF",
        "p95_abs_DeltaF",
        "rms_DeltaF",
        "max_abs_DeltaF",
        "K_at_max_abs_DeltaF",
        "F_intersection_at_max_abs_DeltaF",
        "F_Hopf_at_max_abs_DeltaF",
        "DeltaF_at_max_abs_DeltaF",
    ]
    point_columns = [
        "mode_index",
        "m",
        "n",
        "q",
        "q2",
        "branch_id",
        "root_id",
        "kind_id",
        "curve_id",
        "hopf_branch_id",
        "K",
        "F_intersection",
        "F_Hopf",
        "DeltaF_FHopf_minus_Fintersection",
        "abs_DeltaF",
    ]
    diagnostic_columns = [
        "mode_index",
        "m",
        "n",
        "q",
        "q2",
        "branch_id",
        "root_id",
        "kind_id",
        "n_contour_segments",
        "Z_min",
        "Z_max",
        "Z_span",
        "zero_fraction_absZ_le_tol",
        "skipped_flag",
    ]

    out_mat = outdir / f"{args.prefix}_Hopf_minus_intersection_only.mat"
    savemat(
        out_mat,
        {
            "Kv": Kv.astype(fdtype),
            "Fv": Fv.astype(fdtype),
            "modes_all": modes_all,
            "modes_computed": modes_computed,
            "hopf_table": hopf_table,
            "hopf_table_columns": np.array(["K", "F_Hopf_lower", "F_Hopf_upper"], dtype=object),
            "diff_summary_table": diff_summary_table,
            "diff_summary_columns": np.array(summary_columns, dtype=object),
            "diff_point_table": diff_point_table,
            "diff_point_columns": np.array(point_columns, dtype=object),
            "contour_diagnostic_table": diagnostic_table,
            "contour_diagnostic_columns": np.array(diagnostic_columns, dtype=object),
            "branch_id_meaning": np.array(["+1=v0_plus", "-1=v0_minus"], dtype=object),
            "root_id_meaning": np.array(["+1=lambda_plus", "-1=lambda_minus"], dtype=object),
            "kind_id_meaning": np.array(["1=Re(lambda)=0", "2=Im(lambda)=0"], dtype=object),
            "hopf_branch_id_meaning": np.array(
                ["-1=lower", "0=nearest", "+1=upper"], dtype=object
            ),
            "skipped_flag_meaning": np.array(
                ["0=ok", "1=no finite values", "2=level zero not crossed", "3=almost-flat field"], dtype=object
            ),
            "DeltaF_definition": np.array(
                ["DeltaF = F_Hopf(K) - F_intersection(K)"], dtype=object
            ),
            "Du": np.array([[args.Du]], dtype=np.float64),
            "Dv": np.array([[args.Dv]], dtype=np.float64),
            "Nx": np.array([[args.Nx]], dtype=np.int32),
            "Ny": np.array([[args.Ny]], dtype=np.int32),
            "dx": np.array([[args.dx]], dtype=np.float64),
            "dy": np.array([[args.dy]], dtype=np.float64),
            "params_json": params_json,
        },
        do_compression=True,
    )

    if args.csv:
        write_csv(outdir / f"{args.prefix}_diff_summary.csv", diff_summary_table, summary_columns)
        write_csv(outdir / f"{args.prefix}_contour_diagnostics.csv", diagnostic_table, diagnostic_columns)
        if args.save_point_table:
            write_csv(outdir / f"{args.prefix}_diff_points.csv", diff_point_table, point_columns)

    print("Done.")
    print(f"MAT file: {out_mat}")
    print(f"Summary rows: {diff_summary_table.shape[0]}")
    print(f"Diagnostic rows: {diagnostic_table.shape[0]}")
    print(f"Point rows: {diff_point_table.shape[0]}")


if __name__ == "__main__":
    main()
