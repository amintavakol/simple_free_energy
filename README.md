# Simple Reaction Thermo

A lightweight, self-contained Python pipeline for calculating and visualizing free energy profiles ($\Delta G$) at the semiempirical level of theory for multi-step reaction pathways from SMILES.

---

## Workflow Overview

```mermaid
flowchart TD
    A[Input CSV: Reaction Pathway States] --> B[Step 0: Parse CSV & Identify States]
    B --> C[Step 1: Extract Unique Molecules & Strip Atom Maps]
    C --> C2[Step 1b: Atom Balance Check]
    C2 --> D[Step 2: Generate 3D Seed Conformers via RDKit ETKDGv3]
    D --> E[Step 3: Sample Conformer Ensembles via CREST]
    E --> F[Step 4: GFN2-xTB Optimization & Quasi-RRHO Thermo via tblite/ASE]
    F --> G[Step 5: Boltzmann Aggregation of Conformer Free Energies]
    G --> H[Step 6: Compute ΔG Profiles relative to State 0]
    H --> I[Step 7: Per-Reaction Staircase Plot]
```

Steps 3 and 4 can run **sequentially** (default) or **in parallel** across molecules.

---

## Installation

```bash
pip install rdkit ase tblite matplotlib numpy
```

**Optional:** [CREST](https://github.com/crest-lab/crest) for conformer ensemble sampling. If not found, the pipeline falls back to the single RDKit seed conformer.

---

## Usage

```bash
python fe_pipeline.py reactions.csv --solvent none --crest-binary /path/to/crest --output-dir results/
```

**Parallel execution** (4 molecules simultaneously, 2 cores each — requires 8 cores):
```bash
python fe_pipeline.py reactions.csv --solvent none --crest-binary /path/to/crest --output-dir results/ --parallel --n-workers 4 --n-cores-crest 2
```

**Core budget rule:** `n_workers × n_cores_crest ≤ total available cores`.

---

## Input Format

CSV with one reaction per row. States are separated by `>>`, molecules within a state by `.`. Atom-mapped SMILES are supported and stripped automatically.

```csv
# reaction_id,pathway
RXN000001,"C=CC=C.C=C.[H][H]>>C1=CCCCC1.[H][H]>>C1CCCCC1"
```

Every state must be **atom-balanced** — Step 1b checks this before any QM runs and reports exactly which atoms are missing or gained.

---

## Outputs

- **Per-reaction PNG plots** saved as `{output_dir}/{rxn_id}.png`, one file per reaction
- **Console log** with full per-step diagnostics: atom inventory, optimized energies, vibrational frequencies, Boltzmann weights, and $\Delta G$ profile
- **Optimized geometries** in XYZ format (if `--save-geoms` is set), named by sanitized SMILES + hash

---

## Technical Details

### Atom Balance Check (Step 1b)
Every state is expanded to explicit hydrogens and its atom inventory compared against the previous state. Any discrepancy is reported as a `WARNING` with the exact atoms gained or lost. This is purely diagnostic and does not modify the input — but any energies downstream of an unbalanced step should not be trusted.

### Thermochemistry & Grimme's Quasi-RRHO
Free energy for each conformer is computed as:

$$G = E_{\text{elec}} + \text{ZPVE} + H_{\text{vib}} + H_{\text{trans}} + H_{\text{rot}} + pV - T(S_{\text{vib}} + S_{\text{trans}} + S_{\text{rot}})$$

Vibrational entropy uses Grimme's quasi-RRHO interpolation (Grimme, *Chem. Eur. J.* 2012, 18, 9955):

$$S_{\text{vib}} = w \cdot S_{\text{HO}} + (1 - w) \cdot S_{\text{FR}}, \quad w = \frac{1}{1 + (\omega_0/\nu)^4}$$

Modes below $\omega_0 = 100\ \text{cm}^{-1}$ are smoothly transitioned to a free-rotor model to avoid entropy singularities. Translational and rotational contributions are computed from molecular mass and the inertia tensor. Linear molecules and diatomics (including $H_2$) are handled as a special case with a single moment of inertia.

### Parallel Execution
`ProcessPoolExecutor` dispatches one worker process per unique molecule. Each worker runs Steps 3+4 in full isolation: its own CREST subprocess, its own xTB calculations, and its own temporary directory for ASE `Vibrations` cache files. This last point is critical — without per-worker temp directories, parallel processes write `vib_*.json` files to the same working directory simultaneously, causing index-out-of-bounds errors when frequencies are read back. Results are merged before Step 5, which remains sequential.

### Accuracy Expectations

For organic reactions with GFN2-xTB + quasi-RRHO (no CREST, single conformer):

| Property | Typical error vs experiment |
|---|---|
| Relative conformer energies | ~0.5–1 kcal/mol |
| Reaction $\Delta G$ | ~2–5 kcal/mol |
| Qualitative profile shape | Usually correct |

For improved accuracy: run with CREST to sample conformer ensembles properly, use implicit solvent (ALPB) for any reaction involving polar intermediates or charged species, and consider DFT single-point corrections (e.g. r2SCAN-3c) on the CREST-selected conformers.