# GrayScott Morphological Phase Diagram and Transition

Data and numerical codes supporting the morphological phase diagram, initial-condition robustness analysis, and linear diffusive-mode analysis of the Gray–Scott reaction–diffusion model.

## Overview

This repository contains the numerical data, analysis files, and computational codes associated with a systematic morphological study of the Gray–Scott reaction–diffusion model.

The study examines the organization of dynamical and morphological regimes in the $(F,K)$ parameter plane and, in particular, the robustness of the boundary separating the **active homogeneous** and **temporally variable** regimes under different initial perturbations.

Although the global morphological phase diagram changes substantially when the initial condition is modified, the class $(2\rightarrow3)$ transition boundary is found to remain comparatively robust over a broad common interval of (K).

The repository also contains calculations of the homogeneous Hopf bifurcation and diffusive eigenmodes used to investigate the dynamical origin and location of this morphological transition.

---

## Gray–Scott model

The simulations solve the two-dimensional Gray–Scott reaction–diffusion system

$$
\frac{\partial U}{\partial t}=D_U\nabla^2U-UV^2+F(1-U),$$
$$
\frac{\partial V}{\partial t}=D_V\nabla^2V+UV^2-(F+K)V,$$

where:

* (U(x,y,t)) and (V(x,y,t)) are the two interacting fields,
* (D_U) and (D_V) are diffusion coefficients,
* (F) is the feed parameter,
* (K) is the removal parameter.

The diffusion coefficients used in the morphological parameter sweeps are

$$
D_U=0.16,
\qquad
D_V=0.08.
$$

---

## Numerical protocol

The main morphological phase diagrams were generated over

$$
F,K\in[0,0.10],
$$

using a (250\times250) parameter grid, corresponding to

$$
62,500
$$

simulations for each initial-condition study.

The spatial simulations use:

* grid size: (N_x=N_y=100),
* spatial spacing: (\Delta x=\Delta y=1),
* time step: (\Delta t=1),
* final simulation time: (T=20,000),
* final statistical window: (M=500),
* (D_U=0.16),
* (D_V=0.08),
* homogeneous Neumann boundary conditions implemented using reflected ghost points.

The homogeneous background state is

$$
U=1,\qquad V=0.
$$

---

## Initial conditions

Three perturbations are compared:

1. **Centered perturbation**

$[U_0=0.10,\qquad V_0=0.90]$

2. **Square perturbation**

$[U_0=0.50,\qquad V_0=0.25]$

3. **Square perturbation**

$[U_0=0.10,\qquad V_0=0.90]$

These initial conditions produce visibly different global morphological phase diagrams and therefore provide a direct test of the sensitivity of the phase-space organization to the initial perturbation.

---

## Morphological classification

The parameter plane is organized into seven final morphological classes:

| Class | Morphological regime      |
| ----- | ------------------------- |
| 1     | Saturation                |
| 2     | Active homogeneous        |
| 3     | Temporally variable       |
| 4     | Labyrinths                |
| 5     | Spot division             |
| 6     | Stationary localized spot |
| 7     | Extinction                |

The classification is based on numerical observables obtained from the final simulation window, including:

* mean value of (V),
* maximum value of (V),
* spatial heterogeneity,
* temporal standard deviation of the spatial mean,
* saturated-area fraction,
* number of connected components,
* number of spatial peaks,
* morphological shape descriptors.

An internal default/unclassified label may be used during intermediate computational stages before the ordered classification rules are applied.

---

## Main result: robust class 2–3 boundary

The central comparison concerns the boundary separating:

* **Class 2 — Active homogeneous**, and
* **Class 3 — Temporally variable**.

A refined parameter scan was performed around this transition for the three initial conditions.

Within the common interval

$$
K\in[0.006,,0.05794],
$$

the extracted fronts contain 687 comparison points per initial condition.

Using a refined (F)-grid spacing of approximately

$$
\Delta F_{\mathrm{grid}}=3.30\times10^{-5},
$$

the three independently extracted boundaries show strong agreement:

* 349 points (50.80%) coincide exactly for all three initial conditions,
* 232 points (33.77%) differ by no more than one grid step,
* 71 points (10.33%) differ by no more than two grid steps,
* 94.91% of the comparison points therefore agree within two refined grid steps.

Only a small fraction of the boundary shows larger deviations.

This robustness is local to the class (2\rightarrow3) boundary and should not be interpreted as invariance of the complete morphological phase diagram.

---

# Repository structure

The available material is divided into three principal data groups.

```text
GrayScott-Morphological-Phase-Diagram-and-Transition/
│
├── README.md
│
├── Available_data_part_1_MPD_robust_boundary_organizedHopf/
│   ├── CSV/
│   ├── Codes/
│   └── MAT files/
│
├── Available_data_part_2_MPD_robust_boundary_organizedHopf/
│   └── CSV/
│
└── Available_data_part_3_MPD_robust_boundary_organizedHopf/
    ├── diffusive_modes_000_200_precision/
    ├── diffusive_modes_200_400_precision/
    ├── diffusive_modes_281_330_precision/
    └── diffusive_modes_2000_2200_precision/
```

---

## Part 1 — Morphological phase diagrams and transition boundary

Directory:

```text
Available_data_part_1_MPD_robust_boundary_organizedHopf/
```

This directory contains the main data and codes used for the morphological phase-diagram analysis.

### MAT files

```text
MAT files/
```

contains the three principal morphological phase-diagram datasets:

```text
MPD_centrada_u010_v090_DATA.mat
MPD_cuadrada_u010_v090_DATA.mat
MPD_cuadrada_u050_v025_DATA.mat
```

These files correspond to the three initial perturbations used to test the dependence of the global morphological phase diagram on initial conditions.

---

### Extracted class 2–3 boundaries

The `CSV/` directory contains the extracted transition curves:

```text
Extracted_class_2_3_morphological_boundary_curve_center_u010_v090_ref.csv

Extracted_class_2_3_morphological_boundary_curve_square_u010_v090_ref.csv

Extracted_class_2_3_morphological_boundary_curve_square_u050_v025_ref.csv
```

These files contain the refined class (2\rightarrow3) morphological fronts used for the direct initial-condition comparison.

---

### Region statistics

The same directory contains:

```text
Region_point_counts_7_classes.csv
Region_point_counts_7_classes_summary.csv
```

These files summarize the number of parameter-space points assigned to each morphological regime and allow quantitative comparison of the global phase diagrams.

---

### Refined Hopf-band data

The repository also includes refined parameter sweeps around the lower Hopf branch, including:

```text
Full_parameter_sweep_around_the_Hopf_bifurcation_square_u01_v09_neumann_ref_UNIDO.csv
```

Additional corresponding full-sweep files are contained in Part 2.

---

## Numerical codes

The directory

```text
Available_data_part_1_MPD_robust_boundary_organizedHopf/Codes/
```

contains the principal analysis programs.

### `gray_scott_sweep_MPD.py`

Numerical Gray–Scott parameter sweep and morphological phase-diagram classification.

### `Class 2–3 morphological boundary.py`

Analysis and extraction of the class (2\rightarrow3) morphological transition boundary.

### `gray_scott_hopf_diffusive_modes_deltaF.py`

Calculation and comparison of the morphological boundary with the homogeneous Hopf reference and diffusive-mode information.

### `gray_scott_lambda_q_modal_scan.py`

Linear modal analysis of the Gray–Scott system as a function of spatial wavenumber (q).

### `RECLASSIFICATION_NOTE.md`

Documents a traceable post-processing correction applied to residual points that had originally remained unclassified in one refined transition scan.

Residual points whose temporal variability satisfied

$$
\mathrm{std}_t(\langle V\rangle)\ge10^{-3}
$$

were assigned to the temporally variable class.

The correction modifies only the final morphological label. It does not modify the numerical fields, Gray–Scott integration, initial condition, boundary conditions, or computed observables.

---

## Part 2 — Additional refined Hopf-band datasets

Directory:

```text
Available_data_part_2_MPD_robust_boundary_organizedHopf/
```

contains additional full refined parameter-sweep files:

```text
Full_parameter_sweep_around_the_Hopf_bifurcation_center_u01_v09_neumann_ref_UNIDO.csv

Full_parameter_sweep_around_the_Hopf_bifurcation_square_u050_v025_neumann_ref_UNIDO.csv
```

Together with the corresponding square-((U_0,V_0)=(0.10,0.90)) file stored in Part 1, these datasets provide the refined scans used to compare the transition under all three initial conditions.

---

## Part 3 — Diffusive-mode analysis

Directory:

```text
Available_data_part_3_MPD_robust_boundary_organizedHopf/
```

contains high-precision calculations of linear diffusive modes.

The results are divided into several modal ranges:

```text
diffusive_modes_000_200_precision/
diffusive_modes_200_400_precision/
diffusive_modes_281_330_precision/
diffusive_modes_2000_2200_precision/
```

The stored outputs include `.mat` datasets and CSV summaries describing quantities such as:

```text
*_Hopf_deltaF_precision.mat
*_contour_diagnostics.csv
*_contour_ranges.csv
*_diff_summary.csv
*_mode_status.csv
*_qcrit.csv
```

These calculations are intended to examine the modal structure associated with the homogeneous Hopf reference and its modification for nonzero spatial wavenumber.

---

## Hopf curve

The homogeneous Hopf bifurcation is used as a **dynamical reference**, not as the definition of the morphological transition.

The morphological class (2\rightarrow3) boundary is determined independently from the numerical morphology classifier.

The relative position of the extracted front with respect to the lower Hopf branch can subsequently be characterized through

$$
\Delta F_H
==========

F_{\mathrm{front}}-F_H.
$$

Thus, the Hopf curve provides a useful reference for interpreting the transition but does not replace the morphology-based classification.

---

## Reproducibility

The repository is intended to provide the numerical information required to:

1. reconstruct the morphological phase diagrams,
2. compare the three initial conditions,
3. extract and compare the class (2\rightarrow3) boundaries,
4. reproduce the refined scans around the lower Hopf branch,
5. inspect the seven-class region statistics,
6. reproduce the homogeneous and diffusive linear-stability calculations.

The `.mat` files can be read directly in MATLAB or with compatible Python libraries.

The `.csv` files are plain-text tabular datasets and can be analyzed using MATLAB, Python, R, or equivalent numerical software.

The numerical scripts are written in Python.

---

## Data provenance

The large-scale numerical simulations were performed using computational resources of the **Laboratorio Nacional de Supercómputo del Sureste de México (LNS)** under project:

```text
202504068C
```

The repository contains processed datasets and numerical codes associated with the reported analyses.

---

## Data availability

All data and numerical codes deposited in this repository are provided to support reproducibility and independent inspection of the morphological phase-diagram and transition-boundary analyses.

A permanent archived version of the repository will be deposited in **Zenodo** and associated with a DOI.

The DOI should be cited once the archival release is available.

```text
Zenodo DOI: [to be added]
```

---

## Citation

If you use these data or codes, please cite the associated article and the archived repository version.

```text
Article citation: [to be added after publication / preprint release]

Repository DOI: [to be added after Zenodo archival]
```

A `CITATION.cff` file can also be added to the root of this repository to provide machine-readable citation metadata.

---

## Versioning

The version corresponding to the submitted manuscript will be preserved as a tagged GitHub release and archived through Zenodo.

Future changes to the repository will not alter the archived version associated with the manuscript DOI.

---

## Contact

For questions concerning the data, numerical implementation, or morphological classification, please use the contact information provided in the associated manuscript.

---

## Acknowledgment

Numerical calculations were carried out using computational resources provided by the Laboratorio Nacional de Supercómputo del Sureste de México (LNS), project No. **202504068C**.
