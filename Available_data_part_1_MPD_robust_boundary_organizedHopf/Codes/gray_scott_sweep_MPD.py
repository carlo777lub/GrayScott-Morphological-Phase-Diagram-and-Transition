#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code author: Cruz Castillo Carlos Alberto
Gray--Scott 2D sweep with homogeneous Neumann boundary conditions.
Morphological version v3: separates saturation, Cte_i, spots/proxy-cells,
labyrinths/stripes, and static spot/morphological senescence.

Default configuration for refining the full map:
    NF = 250, NK = 250
    F  in [0.00, 0.10]
    K  in [0.00, 0.10]
    BLOCK_F = 10 -> 25 blocks
    T = 20000
    --array=1-25%2 recommended in SLURM

Metrics saved for each (F,K) pair:
    meanV_map_block
    stdV_map_block
    tstd_map_block
    maxV_map_block
    area_abs_frac_map_block
    area_frac_map_block
    ncomp_raw_map_block
    npeaks_map_block
    largest_comp_domain_frac_map_block
    largest_comp_mask_frac_map_block
    median_area_map_block
    median_circularity_map_block
    spot_density_map_block
    label_map_block

Classes used in the manuscript:
    1 = Saturation / global expansion of V
    2 = Cte_i / homogeneous active state
    3 = Temporally variable / oscillatory
    4 = Labyrinth / stripes
    5 = Spot division
    6 = Stationary localized spot / morphological senescence
    7 = Approximately zero constant state / extinction
    8 = Other / transition / not classified by the 7 manuscript classes

Notes:
    - NF x NK denotes the parameter map, not the spatial grid.
    - The default spatial grid is NX=NY=100.
    - area_abs_frac_map_block is the domain fraction with V >= TH_SAT_V_ABS.
      It is used for saturation and does not depend on a mask relative to the local maximum.
    - Class 6 represents a localized static morphology; by itself, it does not
      demonstrate biological senescence.
    - Class 8 remains a technical/unclassified residual; it is not part
      of the 7 main classes in the manuscript.
    - The output name automatically encodes the actual F and K ranges used.
    - The script does not use scikit-image; only numpy, scipy, and scipy.io.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.io import loadmat, savemat


@dataclass(frozen=True)
class Params:
    Du: float = 0.16
    Dv: float = 0.08
    a2: float = 1.0
    Nx: int = 100
    Ny: int = 100
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 1.0
    T: int = 20000
    M: int = 500
    TH_EXT: float = 0.02
    TH_SPATIAL: float = 0.010
    TH_TEMPORAL: float = 1e-3


@dataclass(frozen=True)
class MorphParams:
    # Final morphological segmentation of V for components and peaks.
    THR_MODE: str = "relative"      # relative | fixed
    TH_REL: float = 0.35            # morphological mask = V >= TH_REL*max(V)
    TH_FIX: float = 0.08            # morphological mask = V >= TH_FIX if THR_MODE=fixed
    MIN_COMPONENT_AREA: int = 6

    # Localized presence and absolute saturation.
    TH_ACTIVE_ABS: float = 0.08     # Vmax >= threshold: localized active structure exists
    TH_SAT_V_ABS: float = 0.80      # absolute saturation area: V >= this threshold
    TH_SAT_MEAN: float = 0.80       # high mean concentration
    TH_SAT_AREA_FRAC: float = 0.65  # domain fraction above TH_SAT_V_ABS

    # Local peaks. They work better than watershed to avoid over-splitting some spots.
    PEAK_MIN_REL: float = 0.45      # peaks with V >= PEAK_MIN_REL*max(V)
    PEAK_PROM_REL: float = 0.05     # minimum local prominence relative to max(V)
    PEAK_FOOTPRINT: int = 5         # local-maximum neighborhood
    MIN_PEAK_AREA: int = 1

    # Spot proliferation.
    TH_SPOT_COUNT: int = 8
    TH_PEAK_COUNT: int = 8
    TH_SPOT_DENSITY: float = 8.0    # objects per 10^4 pixels
    TH_CIRCULARITY_MIN: float = 0.20
    TH_LARGEST_DOMAIN_FRAC_SPOT_MAX: float = 0.025
    TH_LARGEST_MASK_FRAC_SPOT_MAX: float = 0.20

    # Static spot / morphological senescence.
    TH_SEN_NCOMP_MAX: int = 3
    TH_SEN_NPEAK_MAX: int = 3
    TH_SEN_LARGEST_DOMAIN_FRAC_MAX: float = 0.08
    TH_SEN_LARGEST_MASK_FRAC_MIN: float = 0.50

    # Labyrinths / stripes.
    TH_LAB_LARGEST_DOMAIN_FRAC: float = 0.025
    TH_LAB_LARGEST_MASK_FRAC: float = 0.20
    TH_LAB_CIRCULARITY_MAX: float = 0.20


def log(msg: str) -> None:
    print(msg, flush=True)


def getenv_int(name: str, default: int, minimum: int | None = 1) -> int:
    raw = os.getenv(name, "")
    try:
        value = int(float(raw))
        if minimum is not None and value < minimum:
            return default
        return value
    except (TypeError, ValueError):
        return default


def getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def getenv_str(name: str, default: str) -> str:
    raw = os.getenv(name, "")
    return default if raw == "" else raw.strip()


def getenv_bool01(name: str, default: int = 0) -> bool:
    raw = os.getenv(name, "")
    if raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def tag_number(x: float, ndigits: int = 4) -> str:
    """Return a filename-safe numeric tag, for example 0.1000 -> 0p1000."""
    return f"{float(x):.{ndigits}f}".replace("-", "m").replace(".", "p")


def make_run_suffix(nf: int, nk: int, T: int, f_min: float, f_max: float, k_min: float, k_max: float) -> str:
    """Build a suffix that reflects the actual ranges used in the sweep."""
    return (
        f"serial_{nf}x{nk}_T{T}_"
        f"F{tag_number(f_min)}_{tag_number(f_max)}_"
        f"K{tag_number(k_min)}_{tag_number(k_max)}_"
        f"morfologia_v3_senescencia"
    )


def make_params_from_env() -> Params:
    p = Params()
    p_env = Params(
        Du=getenv_float("DU", p.Du),
        Dv=getenv_float("DV", p.Dv),
        a2=getenv_float("A2", p.a2),
        Nx=getenv_int("NX", p.Nx, minimum=3),
        Ny=getenv_int("NY", p.Ny, minimum=3),
        dx=getenv_float("DX", p.dx),
        dy=getenv_float("DY", p.dy),
        dt=getenv_float("DT", p.dt),
        T=getenv_int("T", p.T, minimum=1),
        M=getenv_int("M", p.M, minimum=1),
        TH_EXT=getenv_float("TH_EXT", p.TH_EXT),
        TH_SPATIAL=getenv_float("TH_SPATIAL", p.TH_SPATIAL),
        TH_TEMPORAL=getenv_float("TH_TEMPORAL", p.TH_TEMPORAL),
    )
    if p_env.M > p_env.T:
        log(f"Advertencia: M={p_env.M} era mayor que T={p_env.T}; se ajusta M=T.")
        p_env = replace(p_env, M=p_env.T)
    return p_env


def make_morph_params_from_env() -> MorphParams:
    mp = MorphParams()
    return MorphParams(
        THR_MODE=getenv_str("THR_MODE", mp.THR_MODE),
        TH_REL=getenv_float("TH_REL", mp.TH_REL),
        TH_FIX=getenv_float("TH_FIX", mp.TH_FIX),
        MIN_COMPONENT_AREA=getenv_int("MIN_COMPONENT_AREA", mp.MIN_COMPONENT_AREA, minimum=1),
        TH_ACTIVE_ABS=getenv_float("TH_ACTIVE_ABS", mp.TH_ACTIVE_ABS),
        TH_SAT_V_ABS=getenv_float("TH_SAT_V_ABS", mp.TH_SAT_V_ABS),
        TH_SAT_MEAN=getenv_float("TH_SAT_MEAN", mp.TH_SAT_MEAN),
        TH_SAT_AREA_FRAC=getenv_float("TH_SAT_AREA_FRAC", mp.TH_SAT_AREA_FRAC),
        PEAK_MIN_REL=getenv_float("PEAK_MIN_REL", mp.PEAK_MIN_REL),
        PEAK_PROM_REL=getenv_float("PEAK_PROM_REL", mp.PEAK_PROM_REL),
        PEAK_FOOTPRINT=getenv_int("PEAK_FOOTPRINT", mp.PEAK_FOOTPRINT, minimum=3),
        MIN_PEAK_AREA=getenv_int("MIN_PEAK_AREA", mp.MIN_PEAK_AREA, minimum=1),
        TH_SPOT_COUNT=getenv_int("TH_SPOT_COUNT", mp.TH_SPOT_COUNT, minimum=1),
        TH_PEAK_COUNT=getenv_int("TH_PEAK_COUNT", mp.TH_PEAK_COUNT, minimum=1),
        TH_SPOT_DENSITY=getenv_float("TH_SPOT_DENSITY", mp.TH_SPOT_DENSITY),
        TH_CIRCULARITY_MIN=getenv_float("TH_CIRCULARITY_MIN", mp.TH_CIRCULARITY_MIN),
        TH_LARGEST_DOMAIN_FRAC_SPOT_MAX=getenv_float(
            "TH_LARGEST_DOMAIN_FRAC_SPOT_MAX", mp.TH_LARGEST_DOMAIN_FRAC_SPOT_MAX
        ),
        TH_LARGEST_MASK_FRAC_SPOT_MAX=getenv_float(
            "TH_LARGEST_MASK_FRAC_SPOT_MAX", mp.TH_LARGEST_MASK_FRAC_SPOT_MAX
        ),
        TH_SEN_NCOMP_MAX=getenv_int("TH_SEN_NCOMP_MAX", mp.TH_SEN_NCOMP_MAX, minimum=1),
        TH_SEN_NPEAK_MAX=getenv_int("TH_SEN_NPEAK_MAX", mp.TH_SEN_NPEAK_MAX, minimum=0),
        TH_SEN_LARGEST_DOMAIN_FRAC_MAX=getenv_float(
            "TH_SEN_LARGEST_DOMAIN_FRAC_MAX", mp.TH_SEN_LARGEST_DOMAIN_FRAC_MAX
        ),
        TH_SEN_LARGEST_MASK_FRAC_MIN=getenv_float(
            "TH_SEN_LARGEST_MASK_FRAC_MIN", mp.TH_SEN_LARGEST_MASK_FRAC_MIN
        ),
        TH_LAB_LARGEST_DOMAIN_FRAC=getenv_float(
            "TH_LAB_LARGEST_DOMAIN_FRAC", mp.TH_LAB_LARGEST_DOMAIN_FRAC
        ),
        TH_LAB_LARGEST_MASK_FRAC=getenv_float(
            "TH_LAB_LARGEST_MASK_FRAC", mp.TH_LAB_LARGEST_MASK_FRAC
        ),
        TH_LAB_CIRCULARITY_MAX=getenv_float("TH_LAB_CIRCULARITY_MAX", mp.TH_LAB_CIRCULARITY_MAX),
    )


def adjust_dt_for_stability(p: Params) -> Params:
    dmax = max(p.Du, p.Dv)
    dt_max = (p.dx**2 * p.dy**2) / (2.0 * dmax * (p.dx**2 + p.dy**2))
    safety = 0.90
    if p.dt > safety * dt_max:
        dt_old = p.dt
        dt_new = safety * dt_max
        warnings.warn(
            f"dt={dt_old:.4g} era grande para difusión explícita; "
            f"ajustado a dt={dt_new:.4g} (límite≈{dt_max:.4g})."
        )
        return replace(p, dt=dt_new)
    return p


def laplaciano_neumann_2d(X: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute the 2D Laplacian with homogeneous Neumann conditions using reflected ghost points."""
    nx, ny = X.shape
    Xg = np.empty((nx + 2, ny + 2), dtype=np.float64)
    Xg[1:-1, 1:-1] = X
    Xg[0, 1:-1] = X[1, :]
    Xg[-1, 1:-1] = X[-2, :]
    Xg[1:-1, 0] = X[:, 1]
    Xg[1:-1, -1] = X[:, -2]
    Xg[0, 0] = X[1, 1]
    Xg[0, -1] = X[1, -2]
    Xg[-1, 0] = X[-2, 1]
    Xg[-1, -1] = X[-2, -2]
    return (
        (Xg[2:, 1:-1] - 2.0 * Xg[1:-1, 1:-1] + Xg[:-2, 1:-1]) / dx**2
        + (Xg[1:-1, 2:] - 2.0 * Xg[1:-1, 1:-1] + Xg[1:-1, :-2]) / dy**2
    )


def make_mask(V: np.ndarray, mp: MorphParams) -> np.ndarray:
    vmax = float(np.max(V))
    if vmax <= np.finfo(float).eps:
        return np.zeros_like(V, dtype=bool)
    if mp.THR_MODE.lower() == "fixed":
        return V >= mp.TH_FIX
    return V >= (mp.TH_REL * vmax)


def remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_features = ndimage.label(mask, structure=structure)
    if n_features == 0:
        return mask & False, labeled, np.array([], dtype=int)
    areas = np.bincount(labeled.ravel())[1:]
    valid_ids = np.where(areas >= min_area)[0] + 1
    cleaned = np.isin(labeled, valid_ids)
    labeled_clean, _ = ndimage.label(cleaned, structure=structure)
    return cleaned, labeled_clean, valid_ids


def count_local_peaks(V: np.ndarray, mask: np.ndarray, mp: MorphParams) -> float:
    """Count robust local maxima inside the active mask.

    Plateaus or small noise are not counted by applying two filters:
    (1) minimum height relative to the global maximum of V;
    (2) minimum local prominence inside the PEAK_FOOTPRINT neighborhood.
    """
    vmax = float(np.max(V))
    if vmax <= np.finfo(float).eps or not np.any(mask):
        return 0.0

    size = max(3, int(mp.PEAK_FOOTPRINT))
    if size % 2 == 0:
        size += 1

    local_max = ndimage.maximum_filter(V, size=size, mode="nearest")
    local_min = ndimage.minimum_filter(V, size=size, mode="nearest")
    eps = 1e-12

    peak_mask = (
        (V >= local_max - eps)
        & mask
        & (V >= mp.PEAK_MIN_REL * vmax)
        & ((V - local_min) >= mp.PEAK_PROM_REL * vmax)
    )

    peak_mask, labeled_peaks, _ = remove_small_components(peak_mask, mp.MIN_PEAK_AREA)
    if not np.any(peak_mask):
        return 0.0
    return float(int(labeled_peaks.max()))


def component_metrics(V: np.ndarray, mp: MorphParams) -> Dict[str, float]:
    vmax = float(np.max(V))
    # Absolute fraction for saturation: avoids confusing a uniform Cte_i state with high V.
    area_abs_frac = float(np.mean(V >= mp.TH_SAT_V_ABS))

    mask0 = make_mask(V, mp)
    mask, labeled, _ = remove_small_components(mask0, mp.MIN_COMPONENT_AREA)
    n_pixels = float(mask.size)
    mask_area = float(mask.sum())

    if mask_area <= 0:
        return {
            "maxV": vmax,
            "area_abs_frac": area_abs_frac,
            "area_frac": 0.0,
            "ncomp_raw": 0.0,
            "npeaks": 0.0,
            "largest_domain_frac": 0.0,
            "largest_mask_frac": 0.0,
            "median_area": 0.0,
            "median_circularity": 0.0,
            "spot_density": 0.0,
        }

    ncomp = int(labeled.max())
    areas = np.bincount(labeled.ravel())[1:].astype(float) if ncomp > 0 else np.array([], dtype=float)
    areas = areas[areas > 0]
    largest_area = float(areas.max()) if areas.size else 0.0

    structure = np.ones((3, 3), dtype=bool)
    circularities = []
    for lab_id in range(1, ncomp + 1):
        comp = labeled == lab_id
        area = float(comp.sum())
        if area < mp.MIN_COMPONENT_AREA:
            continue
        eroded = ndimage.binary_erosion(comp, structure=structure, border_value=0)
        boundary = comp & (~eroded)
        perimeter_proxy = float(boundary.sum())
        if perimeter_proxy > 0:
            c = 4.0 * math.pi * area / (perimeter_proxy**2)
            circularities.append(max(0.0, min(1.0, c)))

    npeaks = count_local_peaks(V, mask, mp)
    return {
        "maxV": vmax,
        "area_abs_frac": area_abs_frac,
        "area_frac": mask_area / n_pixels,
        "ncomp_raw": float(ncomp),
        "npeaks": float(npeaks),
        "largest_domain_frac": largest_area / n_pixels,
        "largest_mask_frac": largest_area / mask_area if mask_area > 0 else 0.0,
        "median_area": float(np.median(areas)) if areas.size else 0.0,
        "median_circularity": float(np.median(circularities)) if circularities else 0.0,
        "spot_density": float(ncomp) / n_pixels * 10000.0,
    }


def simulate_one(F: float, k: float, p: Params, mp: MorphParams) -> Tuple[float, ...]:
    U = np.ones((p.Nx, p.Ny), dtype=np.float64)
    V = np.zeros((p.Nx, p.Ny), dtype=np.float64)

    # Centered square perturbation for Nx=Ny=100.
    # MATLAB equivalent: U(30:70,30:70)=0.1; V(30:70,30:70)=0.9.
    # In Python, the upper endpoint is excluded: 29:70 -> indices 29,...,69.
    U[29:70, 29:70] = 0.50
    V[29:70, 29:70] = 0.25

    acc_meanV = 0.0
    acc_stdV = 0.0
    ts = np.zeros(p.M, dtype=np.float64)
    idx = 0

    for t in range(1, p.T + 1):
        Uold = U.copy()
        Vold = V.copy()
        LU = laplaciano_neumann_2d(Uold, p.dx, p.dy)
        LV = laplaciano_neumann_2d(Vold, p.dx, p.dy)
        UV2 = p.a2 * Uold * (Vold**2)
        U = Uold + p.dt * (p.Du * LU - UV2 + F * (1.0 - Uold))
        V = Vold + p.dt * (p.Dv * LV + UV2 - (F + k) * Vold)
        np.clip(U, 0.0, 1.0, out=U)
        np.clip(V, 0.0, 1.0, out=V)
        if t > p.T - p.M:
            mV = float(np.mean(V))
            acc_meanV += mV
            acc_stdV += float(np.std(V, ddof=0))
            ts[idx] = mV
            idx += 1

    denom = max(idx, 1)
    meanV = acc_meanV / denom
    stdV = acc_stdV / denom
    tstdV = float(np.std(ts[:idx], ddof=0)) if idx > 0 else 0.0
    cm = component_metrics(V, mp)

    return (
        meanV,
        stdV,
        tstdV,
        cm["maxV"],
        cm["area_abs_frac"],
        cm["area_frac"],
        cm["ncomp_raw"],
        cm["npeaks"],
        cm["largest_domain_frac"],
        cm["largest_mask_frac"],
        cm["median_area"],
        cm["median_circularity"],
        cm["spot_density"],
    )


def classify(
    meanV: np.ndarray,
    stdV: np.ndarray,
    tstd: np.ndarray,
    maxV: np.ndarray,
    area_abs_frac: np.ndarray,
    area_frac: np.ndarray,
    ncomp: np.ndarray,
    npeaks: np.ndarray,
    largest_domain: np.ndarray,
    largest_mask: np.ndarray,
    circ: np.ndarray,
    spot_density: np.ndarray,
    p: Params,
    mp: MorphParams,
) -> np.ndarray:
    """Classify the final behavior.

    Key changes in v3:
    - Saturation no longer uses the area of a relative mask; it uses
      area_abs_frac = fraction with V >= TH_SAT_V_ABS.
    - Extinction requires a low mean and absence of active structure, so that
      a localized spot with significant Vmax is not lost because its spatial
      mean is small.
    - Class 6 distinguishes a localized spot with few foci and without
      extensive proliferation (morphological proxy for senescence).
    - Class 8 remains the technical/unclassified residual.
    """
    # Class 8 is now the default technical residual:
    # points that do not satisfy any of the 7 manuscript classes.
    label = np.full(meanV.shape, 8.0, dtype=np.float64)

    active_structure = maxV >= mp.TH_ACTIVE_ABS

    # 7 = Approximately zero constant state / extinction.
    ext = (meanV < p.TH_EXT) & (~active_structure)
    label[ext] = 7

    # 1 = Saturation / global expansion of V.
    sat = (~ext) & (
        (meanV >= mp.TH_SAT_MEAN)
        | (area_abs_frac >= mp.TH_SAT_AREA_FRAC)
    )
    label[sat] = 1

    alive = (~ext) & (~sat)

    # 3 = Temporally variable / oscillatory.
    osc = alive & (tstd > p.TH_TEMPORAL)
    label[osc] = 3

    nonosc = alive & (~osc)

    # 2 = Cte_i / homogeneous active state.
    uniform = nonosc & (stdV <= p.TH_SPATIAL)
    label[uniform] = 2

    spatial = nonosc & (~uniform) & (stdV > p.TH_SPATIAL)

    # 6 = Stationary localized spot / morphological senescence.
    senescence = (
        spatial
        & active_structure
        & (ncomp >= 1)
        & (ncomp <= mp.TH_SEN_NCOMP_MAX)
        & (npeaks <= mp.TH_SEN_NPEAK_MAX)
        & (largest_domain <= mp.TH_SEN_LARGEST_DOMAIN_FRAC_MAX)
        & (largest_mask >= mp.TH_SEN_LARGEST_MASK_FRAC_MIN)
    )
    label[senescence] = 6

    # 5 = Spot division.
    spots = (
        spatial
        & (~senescence)
        & (ncomp >= mp.TH_SPOT_COUNT)
        & (npeaks >= mp.TH_PEAK_COUNT)
        & (spot_density >= mp.TH_SPOT_DENSITY)
        & (largest_domain <= mp.TH_LARGEST_DOMAIN_FRAC_SPOT_MAX)
        & (largest_mask <= mp.TH_LARGEST_MASK_FRAC_SPOT_MAX)
        & (circ >= mp.TH_CIRCULARITY_MIN)
    )
    label[spots] = 5

    # 4 = Labyrinth / stripes.
    labyrinth = (
        spatial
        & (~senescence)
        & (~spots)
        & (
            (largest_domain >= mp.TH_LAB_LARGEST_DOMAIN_FRAC)
            | (largest_mask >= mp.TH_LAB_LARGEST_MASK_FRAC)
            | (circ <= mp.TH_LAB_CIRCULARITY_MAX)
            | (npeaks < mp.TH_PEAK_COUNT)
        )
    )
    label[labyrinth] = 4
    return label


def atomic_savemat(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    savemat(tmp, data, do_compression=False, oned_as="row")
    os.replace(tmp, path)


def build_payload(
    arrays: Dict[str, np.ndarray],
    F_vals: np.ndarray,
    k_vals: np.ndarray,
    F_block: np.ndarray,
    idxF_1based: np.ndarray,
    completed_rows: int,
    p: Params,
    mp: MorphParams,
    taskID: int,
    nBlocks: int,
    blockF: int,
    nf: int,
    nk: int,
    f_min: float,
    f_max: float,
    k_min: float,
    k_max: float,
) -> dict:
    payload = dict(arrays)
    payload.update(
        {
            "F_vals": F_vals,
            "k_vals": k_vals,
            "F_block": F_block,
            "idxF": idxF_1based,
            "completed_rows": np.array([[completed_rows]], dtype=np.int64),
            "Du": np.array([[p.Du]], dtype=np.float64),
            "Dv": np.array([[p.Dv]], dtype=np.float64),
            "a2": np.array([[p.a2]], dtype=np.float64),
            "Nx": np.array([[p.Nx]], dtype=np.int64),
            "Ny": np.array([[p.Ny]], dtype=np.int64),
            "dx": np.array([[p.dx]], dtype=np.float64),
            "dy": np.array([[p.dy]], dtype=np.float64),
            "dt": np.array([[p.dt]], dtype=np.float64),
            "T": np.array([[p.T]], dtype=np.int64),
            "M": np.array([[p.M]], dtype=np.int64),
            "TH_EXT": np.array([[p.TH_EXT]], dtype=np.float64),
            "TH_SPATIAL": np.array([[p.TH_SPATIAL]], dtype=np.float64),
            "TH_TEMPORAL": np.array([[p.TH_TEMPORAL]], dtype=np.float64),
            "THR_MODE": np.array([mp.THR_MODE], dtype=object),
            "TH_REL": np.array([[mp.TH_REL]], dtype=np.float64),
            "TH_FIX": np.array([[mp.TH_FIX]], dtype=np.float64),
            "MIN_COMPONENT_AREA": np.array([[mp.MIN_COMPONENT_AREA]], dtype=np.int64),
            "TH_ACTIVE_ABS": np.array([[mp.TH_ACTIVE_ABS]], dtype=np.float64),
            "TH_SAT_V_ABS": np.array([[mp.TH_SAT_V_ABS]], dtype=np.float64),
            "PEAK_MIN_REL": np.array([[mp.PEAK_MIN_REL]], dtype=np.float64),
            "PEAK_PROM_REL": np.array([[mp.PEAK_PROM_REL]], dtype=np.float64),
            "PEAK_FOOTPRINT": np.array([[mp.PEAK_FOOTPRINT]], dtype=np.int64),
            "MIN_PEAK_AREA": np.array([[mp.MIN_PEAK_AREA]], dtype=np.int64),
            "TH_SAT_MEAN": np.array([[mp.TH_SAT_MEAN]], dtype=np.float64),
            "TH_SAT_AREA_FRAC": np.array([[mp.TH_SAT_AREA_FRAC]], dtype=np.float64),
            "TH_SPOT_COUNT": np.array([[mp.TH_SPOT_COUNT]], dtype=np.int64),
            "TH_PEAK_COUNT": np.array([[mp.TH_PEAK_COUNT]], dtype=np.int64),
            "TH_SPOT_DENSITY": np.array([[mp.TH_SPOT_DENSITY]], dtype=np.float64),
            "TH_CIRCULARITY_MIN": np.array([[mp.TH_CIRCULARITY_MIN]], dtype=np.float64),
            "TH_LARGEST_DOMAIN_FRAC_SPOT_MAX": np.array([[mp.TH_LARGEST_DOMAIN_FRAC_SPOT_MAX]], dtype=np.float64),
            "TH_LARGEST_MASK_FRAC_SPOT_MAX": np.array([[mp.TH_LARGEST_MASK_FRAC_SPOT_MAX]], dtype=np.float64),
            "TH_SEN_NCOMP_MAX": np.array([[mp.TH_SEN_NCOMP_MAX]], dtype=np.int64),
            "TH_SEN_NPEAK_MAX": np.array([[mp.TH_SEN_NPEAK_MAX]], dtype=np.int64),
            "TH_SEN_LARGEST_DOMAIN_FRAC_MAX": np.array([[mp.TH_SEN_LARGEST_DOMAIN_FRAC_MAX]], dtype=np.float64),
            "TH_SEN_LARGEST_MASK_FRAC_MIN": np.array([[mp.TH_SEN_LARGEST_MASK_FRAC_MIN]], dtype=np.float64),
            "TH_LAB_LARGEST_DOMAIN_FRAC": np.array([[mp.TH_LAB_LARGEST_DOMAIN_FRAC]], dtype=np.float64),
            "TH_LAB_LARGEST_MASK_FRAC": np.array([[mp.TH_LAB_LARGEST_MASK_FRAC]], dtype=np.float64),
            "TH_LAB_CIRCULARITY_MAX": np.array([[mp.TH_LAB_CIRCULARITY_MAX]], dtype=np.float64),
            "class_codes": np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64),
            "class_names": np.array([
                "1=saturacion",
                "2=Cte_i_estado_activo_homogeneo",
                "3=temporalmente_variable",
                "4=laberintos",
                "5=division_en_manchas",
                "6=mancha_localizada_estacionaria",
                "7=extincion",
                "8=otros_transicion_no_clasificado",
            ], dtype=object),
            "taskID": np.array([[taskID]], dtype=np.int64),
            "nBlocks": np.array([[nBlocks]], dtype=np.int64),
            "blockF": np.array([[blockF]], dtype=np.int64),
            "NF": np.array([[nf]], dtype=np.int64),
            "NK": np.array([[nk]], dtype=np.int64),
            "F_MIN": np.array([[f_min]], dtype=np.float64),
            "F_MAX": np.array([[f_max]], dtype=np.float64),
            "K_MIN": np.array([[k_min]], dtype=np.float64),
            "K_MAX": np.array([[k_max]], dtype=np.float64),
        }
    )
    return payload


def label_from_arrays(arrays: Dict[str, np.ndarray], p: Params, mp: MorphParams) -> np.ndarray:
    return classify(
        arrays["meanV_map_block"],
        arrays["stdV_map_block"],
        arrays["tstd_map_block"],
        arrays["maxV_map_block"],
        arrays["area_abs_frac_map_block"],
        arrays["area_frac_map_block"],
        arrays["ncomp_raw_map_block"],
        arrays["npeaks_map_block"],
        arrays["largest_comp_domain_frac_map_block"],
        arrays["largest_comp_mask_frac_map_block"],
        arrays["median_circularity_map_block"],
        arrays["spot_density_map_block"],
        p,
        mp,
    )


def main() -> None:
    t0 = time.time()
    p = adjust_dt_for_stability(make_params_from_env())
    mp = make_morph_params_from_env()

    # By default, sweep the full F,K square in [0,0.1].
    # For quick tests, these values can be overridden with environment variables.
    nf = getenv_int("NF", 250, minimum=2)
    nk = getenv_int("NK", 250, minimum=2)
    f_min = getenv_float("F_MIN", 0.00)
    f_max = getenv_float("F_MAX", 0.10)
    k_min = getenv_float("K_MIN", 0.00)
    k_max = getenv_float("K_MAX", 0.10)
    if f_max <= f_min:
        raise ValueError(f"F_MAX={f_max} debe ser mayor que F_MIN={f_min}.")
    if k_max <= k_min:
        raise ValueError(f"K_MAX={k_max} debe ser mayor que K_MIN={k_min}.")

    F_vals = np.linspace(f_min, f_max, nf, dtype=np.float64)
    k_vals = np.linspace(k_min, k_max, nk, dtype=np.float64)
    nF, nK = len(F_vals), len(k_vals)

    blockF = getenv_int("BLOCK_F", 10, minimum=1)
    nBlocks = math.ceil(nF / blockF)
    taskID = getenv_int("SLURM_ARRAY_TASK_ID", 1, minimum=1)
    if taskID > nBlocks:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={taskID} excede nBlocks={nBlocks}. "
            f"Con NF={nF} y BLOCK_F={blockF}, usa --array=1-{nBlocks}."
        )

    checkpoint_every = getenv_int("CHECKPOINT_EVERY", 1, minimum=1)
    progress_every_k = getenv_int("PROGRESS_EVERY_K", 100, minimum=1)
    skip_existing = getenv_bool01("SKIP_EXISTING", default=1)

    iStart = (taskID - 1) * blockF + 1
    iEnd = min(taskID * blockF, nF)
    idxF_1based = np.arange(iStart, iEnd + 1, dtype=np.int64)
    F_block = F_vals[idxF_1based - 1]

    outdir = os.getenv("OUTDIR", os.getcwd())
    os.makedirs(outdir, exist_ok=True)
    suffix = make_run_suffix(nf, nk, p.T, f_min, f_max, k_min, k_max)
    outfile = os.path.join(outdir, f"bloque_F_{iStart:04d}_{iEnd:04d}_task_{taskID:03d}_{suffix}.mat")
    checkpoint_file = os.path.join(outdir, f"checkpoint_F_{iStart:04d}_{iEnd:04d}_task_{taskID:03d}_{suffix}.mat")

    if skip_existing and os.path.exists(outfile):
        log("Archivo final ya existe y SKIP_EXISTING=1; no se recalcula el bloque.")
        log(f"Archivo existente: {outfile}")
        return

    shape = (len(idxF_1based), nK)
    arrays = {
        "meanV_map_block": np.full(shape, np.nan, dtype=np.float64),
        "stdV_map_block": np.full(shape, np.nan, dtype=np.float64),
        "tstd_map_block": np.full(shape, np.nan, dtype=np.float64),
        "maxV_map_block": np.full(shape, np.nan, dtype=np.float64),
        "area_abs_frac_map_block": np.full(shape, np.nan, dtype=np.float64),
        "area_frac_map_block": np.full(shape, np.nan, dtype=np.float64),
        "ncomp_raw_map_block": np.full(shape, np.nan, dtype=np.float64),
        "npeaks_map_block": np.full(shape, np.nan, dtype=np.float64),
        "largest_comp_domain_frac_map_block": np.full(shape, np.nan, dtype=np.float64),
        "largest_comp_mask_frac_map_block": np.full(shape, np.nan, dtype=np.float64),
        "median_area_map_block": np.full(shape, np.nan, dtype=np.float64),
        "median_circularity_map_block": np.full(shape, np.nan, dtype=np.float64),
        "spot_density_map_block": np.full(shape, np.nan, dtype=np.float64),
    }
    completed_rows = 0

    if os.path.exists(checkpoint_file):
        log(f"Checkpoint encontrado: {checkpoint_file}")
        ck = loadmat(checkpoint_file)
        for key in arrays:
            if key not in ck:
                raise ValueError(f"Checkpoint incompleto: falta {key}")
            if tuple(ck[key].shape) != shape:
                raise ValueError(f"Checkpoint incompatible para {key}: {ck[key].shape} esperado {shape}")
            arrays[key] = ck[key]
        completed_rows = int(np.ravel(ck.get("completed_rows", [[0]]))[0])
        completed_rows = max(0, min(completed_rows, len(idxF_1based)))
        log(f"Reanudando desde completed_rows = {completed_rows}")

    log("========================================")
    log(f"Barrido Gray--Scott morfologia v2 revisado | {nF}x{nK} | T={p.T} | BLOCK_F={blockF}")
    log(f"Bloque {taskID}/{nBlocks} -> filas F [{iStart}:{iEnd}] ({len(idxF_1based)} filas)")
    log(f"NF x NK             : {nF} x {nK}")
    log(f"F range             : [{f_min}, {f_max}]")
    log(f"K range             : [{k_min}, {k_max}]")
    log(f"Nx x Ny espacial    : {p.Nx} x {p.Ny}")
    log(f"T, M                : {p.T}, {p.M}")
    log("Parametros morfologicos:")
    log(f"  THR_MODE={mp.THR_MODE}, TH_REL={mp.TH_REL}, TH_FIX={mp.TH_FIX}, MIN_COMPONENT_AREA={mp.MIN_COMPONENT_AREA}")
    log(f"  TH_ACTIVE_ABS={mp.TH_ACTIVE_ABS}, TH_SAT_V_ABS={mp.TH_SAT_V_ABS}, TH_SAT_MEAN={mp.TH_SAT_MEAN}, TH_SAT_AREA_FRAC={mp.TH_SAT_AREA_FRAC}")
    log(f"  PEAK_MIN_REL={mp.PEAK_MIN_REL}, PEAK_PROM_REL={mp.PEAK_PROM_REL}, PEAK_FOOTPRINT={mp.PEAK_FOOTPRINT}")
    log(f"  TH_SPOT_COUNT={mp.TH_SPOT_COUNT}, TH_PEAK_COUNT={mp.TH_PEAK_COUNT}, TH_SPOT_DENSITY={mp.TH_SPOT_DENSITY}")
    log(f"  TH_SEN_NCOMP_MAX={mp.TH_SEN_NCOMP_MAX}, TH_SEN_NPEAK_MAX={mp.TH_SEN_NPEAK_MAX}, TH_SEN_LARGEST_DOMAIN_FRAC_MAX={mp.TH_SEN_LARGEST_DOMAIN_FRAC_MAX}, TH_SEN_LARGEST_MASK_FRAC_MIN={mp.TH_SEN_LARGEST_MASK_FRAC_MIN}")
    log(f"OUTDIR              : {outdir}")
    log(f"Archivo final       : {outfile}")
    log(f"Checkpoint          : {checkpoint_file}")
    log("========================================")

    for ii, Fi in enumerate(F_block):
        if ii < completed_rows:
            log(f"  Fila local {ii + 1}/{len(F_block)} | F = {Fi:.10f} -> saltada")
            continue
        row_t0 = time.time()
        log(f"  Fila local {ii + 1}/{len(F_block)} | F = {Fi:.10f}")
        for j, kj in enumerate(k_vals):
            vals = simulate_one(float(Fi), float(kj), p, mp)
            (
                arrays["meanV_map_block"][ii, j],
                arrays["stdV_map_block"][ii, j],
                arrays["tstd_map_block"][ii, j],
                arrays["maxV_map_block"][ii, j],
                arrays["area_abs_frac_map_block"][ii, j],
                arrays["area_frac_map_block"][ii, j],
                arrays["ncomp_raw_map_block"][ii, j],
                arrays["npeaks_map_block"][ii, j],
                arrays["largest_comp_domain_frac_map_block"][ii, j],
                arrays["largest_comp_mask_frac_map_block"][ii, j],
                arrays["median_area_map_block"][ii, j],
                arrays["median_circularity_map_block"][ii, j],
                arrays["spot_density_map_block"][ii, j],
            ) = vals
            if ((j + 1) % progress_every_k == 0) or ((j + 1) == nK):
                log(f"    Columna K {j + 1}/{nK} terminada (K={kj:.10f}, tiempo fila parcial={(time.time()-row_t0)/60:.2f} min)")

        completed_rows = ii + 1
        if (completed_rows % checkpoint_every == 0) or (completed_rows == len(F_block)):
            partial = dict(arrays)
            partial_label = np.full(shape, np.nan, dtype=np.float64)
            partial_label[:completed_rows, :] = label_from_arrays(
                {k: v[:completed_rows, :] for k, v in arrays.items()}, p, mp
            )
            partial["label_map_block"] = partial_label
            payload = build_payload(partial, F_vals, k_vals, F_block, idxF_1based, completed_rows, p, mp,
                                    taskID, nBlocks, blockF, nf, nk, f_min, f_max, k_min, k_max)
            atomic_savemat(checkpoint_file, payload)
            log(f"    Checkpoint guardado tras fila local {completed_rows}/{len(F_block)}")

    arrays["label_map_block"] = label_from_arrays(arrays, p, mp)
    payload = build_payload(arrays, F_vals, k_vals, F_block, idxF_1based, completed_rows, p, mp,
                            taskID, nBlocks, blockF, nf, nk, f_min, f_max, k_min, k_max)
    atomic_savemat(outfile, payload)
    log(f"Bloque guardado en: {outfile}")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        log(f"Checkpoint eliminado: {checkpoint_file}")
    log(f"Tiempo total del bloque: {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR FATAL EN EL SCRIPT PYTHON")
        log(f"Tipo: {type(exc).__name__}")
        log(f"Mensaje: {exc}")
        raise
