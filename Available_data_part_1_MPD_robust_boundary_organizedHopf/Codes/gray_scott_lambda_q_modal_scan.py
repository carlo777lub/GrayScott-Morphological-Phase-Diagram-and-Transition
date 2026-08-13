#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code author: Cruz Castillo Carlos Alberto
gray_scott_lambda_q_modal_scan.py

Modal analysis of the Gray-Scott model with spatial modes q_{mn}.

Computes, on a parameter grid (K,F):
  - the v0+ and v0- branches of the nontrivial equilibrium;
  - lambda+ and lambda- for each v0 branch and each q mode;
  - compact maps of max Re(lambda), the dominant mode, and the dominant branch;
  - separate q=0 and q>0 maps to identify classical Turing instability;
  - summary contours Re(lambda)=0 of the spatial maximum;
  - optionally, full fields for each q mode.

Notes:
  * By default, a compact .mat summary file is saved.
  * --save-fields saves all lambda fields for each q; it may use a lot of disk space.
  * --save-contours saves contours for each q, branch, and root; it can be slow.
  * --save-summary-contours saves only the contours of the compact summary maps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import savemat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gray-Scott modal analysis with spatial q and MATLAB .mat output"
    )

    # Parameter domain: K on the horizontal axis, F on the vertical axis.
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
    p.add_argument(
        "--include-q0",
        action="store_true",
        help="Include the homogeneous mode q=0 in the mode list.",
    )
    p.add_argument(
        "--n-modes",
        type=int,
        default=100,
        help="Number of (m,n) pairs to evaluate, ordered by increasing q.",
    )
    p.add_argument("--mode-start", type=int, default=0)
    p.add_argument("--mode-stop", type=int, default=None)
    p.add_argument(
        "--eps-nontrivial",
        type=float,
        default=1e-14,
        help="Threshold used to exclude F=0 or F+K=0 from the physical nontrivial equilibrium.",
    )

    # Output.
    p.add_argument("--outdir", type=str, default="out_lambda_q")
    p.add_argument("--prefix", type=str, default="GS_lambda_q")
    p.add_argument(
        "--save-fields",
        action="store_true",
        help="Save full lambda fields for each q. This uses a lot of disk space.",
    )
    p.add_argument(
        "--save-contours",
        action="store_true",
        help="Save contours for each q/branch/root. This is time-consuming.",
    )
    p.add_argument(
        "--save-summary-contours",
        action="store_true",
        help="Save Re(lambda)=0 contours from the compact summary maps.",
    )
    p.add_argument("--contour-min-points", type=int, default=5)
    p.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="float32 reduces memory and storage; float64 gives higher precision.",
    )

    return p.parse_args()


def make_modes(
    n_modes: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    q_type: str = "discrete",
    include_q0: bool = False,
) -> np.ndarray:
    """Generate the first n_modes (m,n) pairs, ordered by increasing q."""
    # Initial search box. If it is not large enough, it is expanded automatically.
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


def v0_branches(F: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute v0+ and v0- as complex extensions, plus the real discriminant."""
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

    return v0p, v0m, disc


def lambda_pair(
    F: np.ndarray,
    K: np.ndarray,
    v0: np.ndarray,
    q2: float,
    Du: float,
    Dv: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lambda+ and lambda- for a given v0 branch and q^2."""
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


def update_max(
    max_re: np.ndarray,
    win_mode: np.ndarray | None,
    win_branch: np.ndarray | None,
    win_root: np.ndarray | None,
    candidate: np.ndarray,
    mode_idx: int,
    branch_id: int,
    root_id: int,
    mask: np.ndarray | None = None,
) -> None:
    """Update the maximum Re(lambda) map."""
    re = np.real(candidate)
    valid = np.isfinite(re)
    if mask is not None:
        valid &= mask
    better = valid & (re > max_re)
    max_re[better] = re[better]
    if win_mode is not None:
        win_mode[better] = mode_idx
    if win_branch is not None:
        win_branch[better] = branch_id
    if win_root is not None:
        win_root[better] = root_id


def contour_segments(
    Kv: np.ndarray,
    Fv: np.ndarray,
    Z: np.ndarray,
    level: float = 0.0,
    min_points: int = 5,
) -> List[np.ndarray]:
    """Extract contours as a list of [K,F] arrays. Requires matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Zr = np.asarray(Z, dtype=float)
    finite = np.isfinite(Zr)
    if not np.any(finite):
        return []

    zmin = np.nanmin(Zr)
    zmax = np.nanmax(Zr)
    if not (zmin <= level <= zmax) or zmin == zmax:
        return []

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111)
    try:
        cs = ax.contour(Kv, Fv, Zr, levels=[level])
        segs = []
        for seg in cs.allsegs[0]:
            if seg.shape[0] >= min_points:
                segs.append(np.asarray(seg, dtype=np.float64))
    finally:
        plt.close(fig)
    return segs


def pack_contours_to_table(
    segments: List[np.ndarray],
    mode_idx: int,
    m: int,
    n: int,
    q: float,
    branch_id: int,
    root_id: int,
    kind_id: int,
) -> np.ndarray:
    """Table: [mode,m,n,q,branch,root,kind,curve_id,K,F]."""
    if not segments:
        return np.empty((0, 10), dtype=np.float64)
    rows = []
    for curve_id, seg in enumerate(segments, start=1):
        meta = np.column_stack(
            [
                np.full(seg.shape[0], mode_idx, dtype=float),
                np.full(seg.shape[0], m, dtype=float),
                np.full(seg.shape[0], n, dtype=float),
                np.full(seg.shape[0], q, dtype=float),
                np.full(seg.shape[0], branch_id, dtype=float),
                np.full(seg.shape[0], root_id, dtype=float),
                np.full(seg.shape[0], kind_id, dtype=float),
                np.full(seg.shape[0], curve_id, dtype=float),
                seg[:, 0],
                seg[:, 1],
            ]
        )
        rows.append(meta)
    return np.vstack(rows)


def pack_summary_contours(
    Kv: np.ndarray,
    Fv: np.ndarray,
    maps: List[Tuple[int, str, np.ndarray]],
    min_points: int,
) -> np.ndarray:
    """Summary table: [map_id, curve_id, K, F]."""
    rows = []
    for map_id, _name, Z in maps:
        segs = contour_segments(Kv, Fv, Z, level=0.0, min_points=min_points)
        for curve_id, seg in enumerate(segs, start=1):
            rows.append(
                np.column_stack(
                    [
                        np.full(seg.shape[0], map_id, dtype=float),
                        np.full(seg.shape[0], curve_id, dtype=float),
                        seg[:, 0],
                        seg[:, 1],
                    ]
                )
            )
    return np.vstack(rows) if rows else np.empty((0, 4), dtype=np.float64)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fields_dir = outdir / "fields_by_q"
    if args.save_fields:
        fields_dir.mkdir(exist_ok=True)

    Kv = np.linspace(args.Kmin, args.Kmax, args.nK, dtype=np.float64)
    Fv = np.linspace(args.Fmin, args.Fmax, args.nF, dtype=np.float64)
    K, F = np.meshgrid(Kv, Fv)

    v0p, v0m, disc_v = v0_branches(F, K)

    # Strict physical mask for the real nontrivial equilibrium.
    # Exclude the F=0 line and the F+K=0 point because the derivation assumes v0 != 0.
    physical_v0_mask = (
        (disc_v >= 0.0)
        & (F > args.eps_nontrivial)
        & ((F + K) > args.eps_nontrivial)
        & np.isfinite(np.real(v0p))
        & np.isfinite(np.real(v0m))
    )

    modes_all = make_modes(
        args.n_modes, args.Nx, args.Ny, args.dx, args.dy, args.q_type, args.include_q0
    )
    mode_stop = args.n_modes if args.mode_stop is None else args.mode_stop
    mode_stop = min(mode_stop, modes_all.shape[0])
    mode_start = max(0, args.mode_start)
    modes = modes_all[mode_start:mode_stop]

    fdtype = np.float32 if args.dtype == "float32" else np.float64
    shape = K.shape

    # Compact maps.
    max_re_all = np.full(shape, -np.inf, dtype=fdtype)
    max_re_physical = np.full(shape, -np.inf, dtype=fdtype)
    max_re_q0_all = np.full(shape, -np.inf, dtype=fdtype)
    max_re_q0_physical = np.full(shape, -np.inf, dtype=fdtype)
    max_re_nonhom_all = np.full(shape, -np.inf, dtype=fdtype)
    max_re_nonhom_physical = np.full(shape, -np.inf, dtype=fdtype)

    win_mode_all = np.full(shape, -1, dtype=np.int16)
    win_branch_all = np.zeros(shape, dtype=np.int8)
    win_root_all = np.zeros(shape, dtype=np.int8)

    win_mode_phys = np.full(shape, -1, dtype=np.int16)
    win_branch_phys = np.zeros(shape, dtype=np.int8)
    win_root_phys = np.zeros(shape, dtype=np.int8)

    win_mode_nonhom_phys = np.full(shape, -1, dtype=np.int16)
    win_branch_nonhom_phys = np.zeros(shape, dtype=np.int8)
    win_root_nonhom_phys = np.zeros(shape, dtype=np.int8)

    contour_tables = []
    mode_summary_rows = []

    combo_specs = [("v0p", +1, v0p), ("v0m", -1, v0m)]

    print("=== Revised Gray-Scott lambda(q) scan ===")
    print(f"Grid: nF={args.nF}, nK={args.nK}, total={args.nF*args.nK:,}")
    print(f"F range: [{args.Fmin}, {args.Fmax}], K range: [{args.Kmin}, {args.Kmax}]")
    print(f"Modes requested: {args.n_modes}; computing [{mode_start}, {mode_stop})")
    print(f"save_fields={args.save_fields}, save_contours={args.save_contours}, save_summary_contours={args.save_summary_contours}")
    print(f"Saving to: {outdir.resolve()}")

    for local_i, row in enumerate(modes):
        global_mode_idx = mode_start + local_i
        m = int(row[0])
        n = int(row[1])
        q = float(row[2])
        q2 = float(row[3])
        is_q0 = q2 <= 1e-30
        print(f"Mode {global_mode_idx:04d}: (m,n)=({m},{n}), q={q:.10g}", flush=True)

        fields_to_save: Dict[str, np.ndarray] = {}
        mode_max_all = -np.inf
        mode_max_phys = -np.inf

        for branch_name, branch_id, v0 in combo_specs:
            lam_plus, lam_minus = lambda_pair(F, K, v0, q2, args.Du, args.Dv)

            for root_name, root_id, lam in [
                ("lambda_plus", +1, lam_plus),
                ("lambda_minus", -1, lam_minus),
            ]:
                update_max(max_re_all, win_mode_all, win_branch_all, win_root_all,
                           lam, global_mode_idx, branch_id, root_id)
                update_max(max_re_physical, win_mode_phys, win_branch_phys, win_root_phys,
                           lam, global_mode_idx, branch_id, root_id, mask=physical_v0_mask)

                if is_q0:
                    update_max(max_re_q0_all, None, None, None, lam, global_mode_idx, branch_id, root_id)
                    update_max(max_re_q0_physical, None, None, None, lam, global_mode_idx, branch_id, root_id, mask=physical_v0_mask)
                else:
                    update_max(max_re_nonhom_all, None, None, None, lam, global_mode_idx, branch_id, root_id)
                    update_max(max_re_nonhom_physical, win_mode_nonhom_phys, win_branch_nonhom_phys, win_root_nonhom_phys,
                               lam, global_mode_idx, branch_id, root_id, mask=physical_v0_mask)

                re = np.real(lam)
                if np.any(np.isfinite(re)):
                    mode_max_all = max(mode_max_all, float(np.nanmax(re)))
                re_phys = np.where(physical_v0_mask, re, np.nan)
                if np.any(np.isfinite(re_phys)):
                    mode_max_phys = max(mode_max_phys, float(np.nanmax(re_phys)))

                if args.save_contours:
                    seg_re = contour_segments(Kv, Fv, np.real(lam), level=0.0, min_points=args.contour_min_points)
                    tab_re = pack_contours_to_table(seg_re, global_mode_idx, m, n, q, branch_id, root_id, kind_id=1)
                    if tab_re.size:
                        contour_tables.append(tab_re)

                    seg_im = contour_segments(Kv, Fv, np.imag(lam), level=0.0, min_points=args.contour_min_points)
                    tab_im = pack_contours_to_table(seg_im, global_mode_idx, m, n, q, branch_id, root_id, kind_id=2)
                    if tab_im.size:
                        contour_tables.append(tab_im)

            if args.save_fields:
                cdtype = np.complex64 if args.dtype == "float32" else np.complex128
                fields_to_save[f"lambda_plus_{branch_name}"] = lam_plus.astype(cdtype)
                fields_to_save[f"lambda_minus_{branch_name}"] = lam_minus.astype(cdtype)
                fields_to_save[f"Re_lambda_plus_{branch_name}"] = np.real(lam_plus).astype(fdtype)
                fields_to_save[f"Im_lambda_plus_{branch_name}"] = np.imag(lam_plus).astype(fdtype)
                fields_to_save[f"Re_lambda_minus_{branch_name}"] = np.real(lam_minus).astype(fdtype)
                fields_to_save[f"Im_lambda_minus_{branch_name}"] = np.imag(lam_minus).astype(fdtype)

        mode_summary_rows.append([global_mode_idx, m, n, q, q2, mode_max_all, mode_max_phys])

        if args.save_fields:
            fname = fields_dir / f"{args.prefix}_fields_mode{global_mode_idx:04d}_m{m:03d}_n{n:03d}.mat"
            savemat(
                fname,
                {
                    "Kv": Kv,
                    "Fv": Fv,
                    "mode_index": np.array([[global_mode_idx]], dtype=np.int32),
                    "m": np.array([[m]], dtype=np.int32),
                    "n": np.array([[n]], dtype=np.int32),
                    "q": np.array([[q]], dtype=np.float64),
                    "q2": np.array([[q2]], dtype=np.float64),
                    "Du": np.array([[args.Du]], dtype=np.float64),
                    "Dv": np.array([[args.Dv]], dtype=np.float64),
                    **fields_to_save,
                },
                do_compression=True,
            )

    for arr in [
        max_re_all, max_re_physical, max_re_q0_all, max_re_q0_physical,
        max_re_nonhom_all, max_re_nonhom_physical,
    ]:
        arr[~np.isfinite(arr)] = np.nan

    # Classical Turing criterion: stable q=0 and at least one unstable q>0, on the real nontrivial equilibrium.
    turing_mask_physical = (
        physical_v0_mask
        & np.isfinite(max_re_q0_physical)
        & np.isfinite(max_re_nonhom_physical)
        & (max_re_q0_physical < 0.0)
        & (max_re_nonhom_physical > 0.0)
    )

    contour_table = np.vstack(contour_tables).astype(np.float64) if contour_tables else np.empty((0, 10), dtype=np.float64)
    mode_summary = np.asarray(mode_summary_rows, dtype=np.float64)

    # Saddle-node curves of the v0 discriminant: F^2 - 4F(F+K)^2 = 0.
    NKsn = 2000
    Ksn = np.linspace(max(0.0, args.Kmin), min(args.Kmax, 1.0 / 16.0), NKsn)
    radSN = 1.0 - 16.0 * Ksn
    radSN[radSN < 0] = np.nan
    sSN = np.sqrt(radSN)
    Fsn_up = (1.0 - 8.0 * Ksn + sSN) / 8.0
    Fsn_lo = (1.0 - 8.0 * Ksn - sSN) / 8.0

    summary_maps = [
        (1, "max_Re_lambda_all", max_re_all),
        (2, "max_Re_lambda_physical", max_re_physical),
        (3, "max_Re_lambda_nonhom_all", max_re_nonhom_all),
        (4, "max_Re_lambda_nonhom_physical", max_re_nonhom_physical),
        (5, "max_Re_lambda_q0_physical", max_re_q0_physical),
    ]
    summary_contour_table = (
        pack_summary_contours(Kv, Fv, summary_maps, args.contour_min_points)
        if args.save_summary_contours else np.empty((0, 4), dtype=np.float64)
    )

    params_json = json.dumps(vars(args).copy(), indent=2)
    summary_name = outdir / f"{args.prefix}_summary.mat"
    savemat(
        summary_name,
        {
            "Kv": Kv,
            "Fv": Fv,
            "modes_all": modes_all,
            "modes_computed": modes,
            "mode_summary": mode_summary,
            "mode_summary_columns": np.array(
                ["mode_index", "m", "n", "q", "q2", "max_Re_all", "max_Re_physical"], dtype=object
            ),
            "disc_v": disc_v.astype(fdtype),
            "physical_v0_mask": physical_v0_mask.astype(np.uint8),
            "v0p": v0p.astype(np.complex64 if args.dtype == "float32" else np.complex128),
            "v0m": v0m.astype(np.complex64 if args.dtype == "float32" else np.complex128),
            "max_Re_lambda_all": max_re_all.astype(fdtype),
            "winner_mode_all": win_mode_all,
            "winner_branch_all": win_branch_all,
            "winner_root_all": win_root_all,
            "max_Re_lambda_physical": max_re_physical.astype(fdtype),
            "winner_mode_physical": win_mode_phys,
            "winner_branch_physical": win_branch_phys,
            "winner_root_physical": win_root_phys,
            "max_Re_lambda_q0_all": max_re_q0_all.astype(fdtype),
            "max_Re_lambda_q0_physical": max_re_q0_physical.astype(fdtype),
            "max_Re_lambda_nonhom_all": max_re_nonhom_all.astype(fdtype),
            "max_Re_lambda_nonhom_physical": max_re_nonhom_physical.astype(fdtype),
            "winner_mode_nonhom_physical": win_mode_nonhom_phys,
            "winner_branch_nonhom_physical": win_branch_nonhom_phys,
            "winner_root_nonhom_physical": win_root_nonhom_phys,
            "turing_mask_physical": turing_mask_physical.astype(np.uint8),
            "contour_table": contour_table,
            "contour_table_columns": np.array(
                ["mode_index", "m", "n", "q", "branch_id", "root_id", "kind_id", "curve_id", "K", "F"], dtype=object
            ),
            "summary_contour_table": summary_contour_table,
            "summary_contour_table_columns": np.array(["map_id", "curve_id", "K", "F"], dtype=object),
            "summary_contour_map_meaning": np.array(
                [
                    "1=max_Re_lambda_all",
                    "2=max_Re_lambda_physical",
                    "3=max_Re_lambda_nonhom_all",
                    "4=max_Re_lambda_nonhom_physical",
                    "5=max_Re_lambda_q0_physical",
                ], dtype=object
            ),
            "branch_id_meaning": np.array(["+1=v0_plus", "-1=v0_minus"], dtype=object),
            "root_id_meaning": np.array(["+1=lambda_plus", "-1=lambda_minus"], dtype=object),
            "kind_id_meaning": np.array(["1=Re(lambda)=0", "2=Im(lambda)=0"], dtype=object),
            "Ksn": Ksn,
            "Fsn_up": Fsn_up,
            "Fsn_lo": Fsn_lo,
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

    print("Done.")
    print(f"Summary file: {summary_name}")
    if args.save_fields:
        print(f"Fields directory: {fields_dir}")


if __name__ == "__main__":
    main()
