"""
fe_pipeline.py
==============
Modular free energy profile pipeline for reaction pathways.

Pipeline
--------
  Step 0   -- Parse CSV
  Step 1   -- Extract unique molecules
  Step 1b  -- Atom balance check
  Step 2   -- RDKit 3D generation
  Step 3   -- CREST conformer sampling
  Step 4   -- GFN2-xTB opt + quasi-RRHO thermo
  Step 5   -- Boltzmann aggregation
  Step 6   -- dG profile per reaction
  Step 7   -- Staircase plot

CSV format
----------
  RXN000001,"CCO>>CC[O-]>>CC=O"
  (states separated by >>  |  molecules within a state separated by .)
"""

import csv
import hashlib
import os
import re
import sys
import subprocess
import tempfile
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from ase import Atoms as AseAtoms
    from ase.io import read as ase_read, write as ase_write
    from ase.optimize import LBFGS
    from ase.vibrations import Vibrations
    from tblite.ase import TBLite as XTB
    ASE_AVAILABLE = True
except ImportError as e:
    ASE_AVAILABLE = False
    print(f"[IMPORT WARNING] ASE/tblite not fully available: {e}")

# -- Physical constants (SI) --------------------------------------------------
_h   = 6.62607015e-34
_kB  = 1.380649e-23
_NA  = 6.02214076e23
_c   = 2.99792458e10
_R   = 8.314462
_cal = 4.184

EV_TO_KCAL = 23.0605
TEMPERATURE = 298.15
PRESSURE    = 101325.0

SEP = "-" * 60


# =============================================================================
#  STEP 0 -- Parse CSV
# =============================================================================

def step0_parse_csv(csv_path):
    """
    Input : path to CSV file
    Output: list of reaction dicts
    """
    print("\n" + SEP)
    print("  STEP 0 -- Parse CSV")
    print(SEP)
    print(f"  INPUT : {csv_path}")

    reactions = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rxn_id  = row[0].strip()
            rxn_str = row[1].strip().strip('"')
            raw_states = rxn_str.split(">>")
            states = []
            for i, state_str in enumerate(raw_states):
                raw_mols = _split_state_smiles(state_str.strip())
                states.append({"label": f"State_{i}", "raw_smiles": raw_mols})
            reactions.append({"rxn_id": rxn_id, "states": states})

    print(f"\n  OUTPUT: {len(reactions)} reaction(s) parsed")
    for rxn in reactions:
        print(f"\n  [{rxn['rxn_id']}]")
        for s in rxn["states"]:
            print(f"    {s['label']:10s} -> {s['raw_smiles']}")

    return reactions


def _split_state_smiles(state_smiles):
    mol = Chem.MolFromSmiles(state_smiles)
    if mol is None:
        raise ValueError(f"Cannot parse state SMILES: {state_smiles!r}")
    frags = Chem.GetMolFrags(mol, asMols=True)
    return [Chem.MolToSmiles(f) for f in frags]


# =============================================================================
#  STEP 1 -- Extract unique canonical SMILES
# =============================================================================

def step1_extract_unique_molecules(reactions):
    """
    Input : reactions from Step 0
    Output: sorted list of unique canonical SMILES (atom maps stripped)
    """
    print("\n" + SEP)
    print("  STEP 1 -- Extract unique molecules")
    print(SEP)
    print(f"  INPUT : {len(reactions)} reaction(s)")

    seen = set()
    for rxn in reactions:
        for state in rxn["states"]:
            for smi in state["raw_smiles"]:
                seen.add(_canonical(smi))

    unique = sorted(seen)

    for rxn in reactions:
        for state in rxn["states"]:
            state["smiles"] = [_canonical(s) for s in state["raw_smiles"]]

    print(f"\n  OUTPUT: {len(unique)} unique molecule(s)")
    for i, smi in enumerate(unique):
        print(f"    [{i}] {smi}")

    return unique


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


# =============================================================================
#  STEP 1b -- Atom balance checker
# =============================================================================

def step1b_check_atom_balance(reactions):
    """
    Input : reactions (after Step 1, so each state has a 'smiles' key)
    Output: True if all reactions are balanced, False otherwise.

    Computes the total atom inventory of each state (including implicit H)
    and compares consecutive states. Reports exactly which atoms are gained
    or lost between states. Purely diagnostic -- does not modify reactions.
    """
    print("\n" + SEP)
    print("  STEP 1b -- Atom balance check")
    print(SEP)

    all_balanced = True

    for rxn in reactions:
        rxn_id = rxn["rxn_id"]
        print(f"\n  [{rxn_id}]")

        # Compute atom inventory per state (explicit H included)
        state_inventories = []
        for state in rxn["states"]:
            inventory = {}
            for smi in state["smiles"]:
                mol = Chem.AddHs(Chem.MolFromSmiles(smi))
                for atom in mol.GetAtoms():
                    elem = atom.GetSymbol()
                    inventory[elem] = inventory.get(elem, 0) + 1
            state_inventories.append(inventory)

            formula = "".join(
                f"{e}{n}" if n > 1 else e
                for e, n in sorted(inventory.items())
            )
            mol_str = " + ".join(state["smiles"])
            print(f"    {state['label']:12s}  {formula:25s}  ({mol_str})")

        # Compare consecutive states
        rxn_balanced = True
        for i in range(1, len(state_inventories)):
            prev = state_inventories[i - 1]
            curr = state_inventories[i]
            all_elems = set(prev) | set(curr)
            diff = {
                e: curr.get(e, 0) - prev.get(e, 0)
                for e in all_elems
                if curr.get(e, 0) != prev.get(e, 0)
            }
            if diff:
                rxn_balanced = False
                all_balanced = False
                gained = {e: n  for e, n in diff.items() if n > 0}
                lost   = {e: -n for e, n in diff.items() if n < 0}
                parts = []
                if lost:
                    s = " ".join(f"{n}{e}" if n > 1 else e for e, n in lost.items())
                    parts.append(f"lost {s}")
                if gained:
                    s = " ".join(f"{n}{e}" if n > 1 else e for e, n in gained.items())
                    parts.append(f"gained {s}")
                print(f"    WARNING: State_{i-1} -> State_{i} UNBALANCED: {', '.join(parts)}")
            else:
                print(f"    OK     : State_{i-1} -> State_{i} balanced")

        if rxn_balanced:
            print(f"    PASS: {rxn_id} is fully atom-balanced")

    if all_balanced:
        print("\n  OUTPUT: all reactions balanced")
    else:
        print("\n  OUTPUT: IMBALANCES DETECTED -- fix CSV before trusting energies")

    return all_balanced


# =============================================================================
#  STEP 2 -- RDKit 3D seed generation
# =============================================================================

def step2_generate_3d(unique_smiles):
    """
    Input : list of canonical SMILES
    Output: {smiles: rdkit_mol_with_3d_conformer}
    """
    print("\n" + SEP)
    print("  STEP 2 -- RDKit 3D seed generation")
    print(SEP)
    print(f"  INPUT : {len(unique_smiles)} SMILES")

    mol_3d = {}
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            warnings.warn(f"ETKDGv3 failed for {smi!r}, trying ETKDG.")
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        ff = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol)
        )
        if ff:
            ff.Minimize(maxIts=2000)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

        mol_3d[smi] = mol
        conf   = mol.GetConformer()
        coords = conf.GetPositions()
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        print(f"\n  [{smi}]")
        print(f"    heavy atoms : {n_heavy}")
        print(f"    total atoms : {mol.GetNumAtoms()}")
        print(f"    centroid    : ({coords[:,0].mean():.2f}, "
              f"{coords[:,1].mean():.2f}, {coords[:,2].mean():.2f}) A")

    print(f"\n  OUTPUT: 3D mols for {len(mol_3d)} molecule(s)")
    return mol_3d


# =============================================================================
#  STEP 3 -- CREST conformer sampling
# =============================================================================

def step3_crest_sampling(mol_3d_dict, solvent="water", n_cores=4,
                         crest_binary="crest"):
    """
    Input : {smiles: rdkit_mol}
    Output: {smiles: [ase.Atoms, ...]}
    """
    print("\n" + SEP)
    print("  STEP 3 -- CREST conformer sampling")
    print(SEP)
    print(f"  INPUT : {len(mol_3d_dict)} molecule(s), solvent={solvent!r}")

    crest_available = _check_binary(crest_binary)
    if not crest_available:
        print(f"  WARNING: '{crest_binary}' not found. Falling back to RDKit seed.")

    ensembles = {}

    for smi, rdmol in mol_3d_dict.items():
        charge, mult = _charge_and_mult(rdmol)
        print(f"\n  [{smi}]  charge={charge}  mult={mult}")

        if not ASE_AVAILABLE:
            print("    ASE not available -- storing RDKit mol as placeholder.")
            ensembles[smi] = [rdmol]
            continue

        seed_atoms = _rdkit_to_ase(rdmol)

        if not crest_available:
            print("    -> single seed conformer (no CREST)")
            ensembles[smi] = [seed_atoms]
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_in = os.path.join(tmpdir, "input.xyz")
            ase_write(xyz_in, seed_atoms)

            cmd = [
                crest_binary, xyz_in,
                "--gfn2",
                "--T", str(n_cores),
                "--chrg", str(charge),
                "--uhf",  str(mult - 1),
                "-quick",
            ]
            if solvent.lower() != "none":
                cmd += ["--alpb", solvent]

            print(f"    CMD: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)

            ensemble_file = os.path.join(tmpdir, "crest_conformers.xyz")
            if result.returncode != 0 or not os.path.exists(ensemble_file):
                print(f"    CREST failed (rc={result.returncode}). Using seed only.")
                print(f"    STDERR: {result.stderr[-300:]}")
                ensembles[smi] = [seed_atoms]
            else:
                conformers = ase_read(ensemble_file, index=":")
                ensembles[smi] = conformers
                print(f"    -> {len(conformers)} conformer(s) from CREST")

    print(f"\n  OUTPUT: ensembles for {len(ensembles)} molecule(s)")
    for smi, confs in ensembles.items():
        n = len(confs) if not isinstance(confs[0], Chem.rdchem.Mol) else "N/A"
        print(f"    {smi:40s} -> {n} conformer(s)")

    return ensembles


def _check_binary(name):
    return subprocess.run(["which", name], capture_output=True).returncode == 0


def _rdkit_to_ase(mol):
    conf      = mol.GetConformer()
    positions = conf.GetPositions()
    symbols   = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return AseAtoms(symbols=symbols, positions=positions)


def _charge_and_mult(mol):
    charge = Chem.GetFormalCharge(mol)
    n_rad  = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return charge, n_rad + 1


def _sanitize_filename(smi):
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', smi)
    if len(clean) > 30:
        clean = clean[:30]
    smi_hash = hashlib.md5(smi.encode('utf-8')).hexdigest()[:6]
    return f"{clean}_{smi_hash}"


# =============================================================================
#  STEP 4 -- GFN2-xTB optimisation + quasi-RRHO thermochemistry
# =============================================================================

def step4_xtb_optimize_and_thermo(ensembles, solvent="water",
                                   temperature=TEMPERATURE, pressure=PRESSURE,
                                   fmax=0.01, vib_delta=0.01,
                                   output_dir=None, save_geoms=False,
                                   vib_dir=None):
    # vib_dir: base directory for Vibrations cache files.
    # In parallel mode each worker passes its own tempdir to avoid
    # race conditions between processes writing to the same CWD.
    # Returns (thermo_results, mol_failures) where mol_failures is a
    # dict {smiles: reason_string} for molecules that could not be computed.
    """
    Input : {smiles: [ase.Atoms, ...]}
    Output: {smiles: [{G, H, S_total, E_elec, ZPE, ...}, ...]}
    """
    print("\n" + SEP)
    print("  STEP 4 -- GFN2-xTB opt + quasi-RRHO thermochemistry")
    print(SEP)
    print(f"  INPUT : {len(ensembles)} molecule(s), solvent={solvent!r}, T={temperature} K")

    if not ASE_AVAILABLE:
        print("  ERROR: ASE/tblite not available. Cannot run Step 4.")
        return {}, {"ALL": "ASE/tblite not available"}

    thermo_results = {}
    mol_failures   = {}   # {smiles: reason}

    for smi, conformers in ensembles.items():
        charge, mult = _get_charge_mult_from_smi(smi)
        print(f"\n  [{smi}]  charge={charge}  mult={mult}  conformers={len(conformers)}")
        mol_thermo = []

        for i, atoms in enumerate(conformers):
            if isinstance(atoms, Chem.rdchem.Mol):
                print(f"    Conformer {i}: RDKit placeholder -- skipping xTB.")
                mol_failures[smi] = "ASE not available at runtime; RDKit placeholder used"
                continue

            print(f"\n    -- Conformer {i} --")

            try:
                atoms = atoms.copy()
                xtb_kwargs = dict(method="GFN2-xTB", charge=charge, multiplicity=mult)
                if solvent.lower() != "none":
                    xtb_kwargs["solvent"] = solvent
                atoms.calc = XTB(**xtb_kwargs)

                opt = LBFGS(atoms, logfile=None)
                opt.run(fmax=fmax)

                if save_geoms and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    smi_clean = _sanitize_filename(smi)
                    geom_filename = f"{smi_clean}_conf{i}.xyz"
                    geom_filepath = os.path.join(output_dir, geom_filename)
                    ase_write(geom_filepath, atoms)
                    print(f"      Optimized geometry saved to {geom_filepath}")

                E_elec_ev = atoms.get_potential_energy()
                print(f"      E_elec   = {E_elec_ev:.6f} eV  ({E_elec_ev * EV_TO_KCAL:.4f} kcal/mol)")

                # Unique name per molecule+conformer to avoid stale cache
                # conflicts when different molecules share the same index (vib_0).
                # In parallel mode vib_dir is a per-worker tempdir so processes
                # never write to the same path simultaneously.
                vib_name = f"vib_{_sanitize_filename(smi)}_{i}"
                if vib_dir:
                    vib_name = os.path.join(vib_dir, vib_name)
                vib = Vibrations(atoms, delta=vib_delta, name=vib_name)
                try:
                    vib.clean()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vib.run()
                    raw_freqs = vib.get_frequencies()
                finally:
                    vib.clean()

                real_freqs = [abs(f.real) for f in raw_freqs
                              if abs(f.imag) < 1.0 and f.real > 10.0]
                n_imag = sum(1 for f in raw_freqs if f.real < -10.0)

                print(f"      frequencies: {len(real_freqs)} real  |  {n_imag} imaginary")
                if real_freqs:
                    print(f"      freq range : {min(real_freqs):.1f} - {max(real_freqs):.1f} cm-1")

                thermo = _quasi_rrho_thermo(atoms, E_elec_ev, real_freqs,
                                            temperature, pressure)
                mol_thermo.append(thermo)

                print(f"      ZPE      = {thermo['ZPE']:.4f} kcal/mol")
                print(f"      H        = {thermo['H']:.4f} kcal/mol")
                print(f"      S*T      = {thermo['S_total'] * temperature:.4f} kcal/mol")
                print(f"      G        = {thermo['G']:.4f} kcal/mol")

            except Exception as exc:
                reason = f"xTB failed for conformer {i}: {type(exc).__name__}: {exc}"
                print(f"      ERROR: {reason}")
                mol_failures[smi] = reason

        thermo_results[smi] = mol_thermo

    n_failed = len(mol_failures)
    print(f"\n  OUTPUT: thermo data for {len(thermo_results)} molecule(s)"
          f"  |  {n_failed} failure(s)")
    if n_failed:
        for smi, reason in mol_failures.items():
            print(f"    FAILED: {smi}  ->  {reason}")
    return thermo_results, mol_failures


def _get_charge_mult_from_smi(smi):
    mol    = Chem.AddHs(Chem.MolFromSmiles(smi))
    charge = Chem.GetFormalCharge(mol)
    n_rad  = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return charge, n_rad + 1


def _quasi_rrho_thermo(atoms, E_elec_ev, freqs_cm1, temperature, pressure,
                        omega0=100.0, symmetry_number=1):
    """
    Compute G (kcal/mol) with Grimme's quasi-RRHO correction.
    Reference: Grimme, Chem. Eur. J. 2012, 18, 9955.
    """
    E_kcal = E_elec_ev * EV_TO_KCAL

    # Zero-point energy
    ZPE = sum(0.5 * _h * _c * nu for nu in freqs_cm1) * _NA / (_cal * 1000)

    # Vibrational enthalpy
    H_vib = 0.0
    for nu in freqs_cm1:
        u = _h * _c * nu / (_kB * temperature)
        H_vib += _R * temperature * u / (np.exp(u) - 1)
    H_vib /= (_cal * 1000)

    # Quasi-RRHO vibrational entropy
    B_av  = 1e-44
    S_vib = 0.0
    for nu in freqs_cm1:
        w    = 1.0 / (1.0 + (omega0 / nu) ** 4)
        u    = _h * _c * nu / (_kB * temperature)
        S_HO = _R * (u / (np.exp(u) - 1) - np.log(1 - np.exp(-u)))
        mu_nu    = _h / (8 * np.pi**2 * _c * nu)
        mu_prime = mu_nu * B_av / (mu_nu + B_av)
        S_FR = _R * (0.5 + np.log(
            np.sqrt(8 * np.pi**3 * mu_prime * _kB * temperature / _h**2)
        ))
        S_vib += w * S_HO + (1 - w) * S_FR
    S_vib /= (_cal * 1000)

    # Translational
    mass_kg = sum(atoms.get_masses()) * 1.66054e-27
    S_trans = _R * (
        np.log((2 * np.pi * mass_kg * _kB * temperature / _h**2) ** 1.5
               * _kB * temperature / pressure)
        + 2.5
    )
    H_trans  = 1.5 * _R * temperature
    S_trans /= (_cal * 1000)
    H_trans /= (_cal * 1000)

    # Rotational
    n_atoms = len(atoms)
    if n_atoms == 1:
        H_rot, S_rot = 0.0, 0.0
    else:
        masses = atoms.get_masses() * 1.66054e-27
        pos    = atoms.get_positions() * 1e-10
        com    = np.average(pos, weights=masses, axis=0)
        pos   -= com
        I = np.zeros((3, 3))
        for m, r in zip(masses, pos):
            I[0,0] += m * (r[1]**2 + r[2]**2)
            I[1,1] += m * (r[0]**2 + r[2]**2)
            I[2,2] += m * (r[0]**2 + r[1]**2)
            I[0,1] -= m * r[0] * r[1]
            I[0,2] -= m * r[0] * r[2]
            I[1,2] -= m * r[1] * r[2]
        I[1,0] = I[0,1]; I[2,0] = I[0,2]; I[2,1] = I[1,2]
        eigvals = np.sort(np.linalg.eigvalsh(I))
        # Use 1e-50 threshold to handle light diatomics like H2
        # whose moment of inertia (~4.6e-48 kg.m2) falls below 1e-47
        nonzero = eigvals[eigvals > 1e-50]
        # 2-atom molecules are always linear; also catch collinear N-atom cases
        is_linear = (n_atoms == 2) or (len(nonzero) <= 1)
        if is_linear:
            # Pick the largest eigenvalue as the single moment of inertia
            IA = nonzero[-1] if len(nonzero) > 0 else eigvals[-1]
            S_rot = _R * (np.log(8 * np.pi**2 * IA * _kB * temperature
                                  / (symmetry_number * _h**2)) + 1.0)
            H_rot = _R * temperature
        else:
            IA, IB, IC = nonzero[0], nonzero[1], nonzero[2]
            S_rot = _R * (0.5 * np.log(
                np.pi * IA * IB * IC / symmetry_number**2
                * (8 * np.pi**2 * _kB * temperature / _h**2) ** 3
            ) + 1.5)
            H_rot = 1.5 * _R * temperature
        H_rot /= (_cal * 1000)
        S_rot /= (_cal * 1000)

    pV      = _R * temperature / (_cal * 1000)
    H_total = E_kcal + ZPE + H_vib + H_trans + H_rot + pV
    S_total = S_vib + S_trans + S_rot
    G       = H_total - temperature * S_total

    return {
        "E_elec":  E_kcal,
        "ZPE":     ZPE,
        "H_vib":   H_vib,
        "H_trans": H_trans,
        "H_rot":   H_rot,
        "H":       H_total,
        "S_vib":   S_vib,
        "S_trans": S_trans,
        "S_rot":   S_rot,
        "S_total": S_total,
        "G":       G,
    }


# =============================================================================
#  STEP 5 -- Boltzmann aggregation
# =============================================================================

def step5_boltzmann_aggregate(thermo_results, temperature=TEMPERATURE):
    """
    Input : {smiles: [{G, ...}, ...]}
    Output: {smiles: G_eff}  (kcal/mol)

    G_eff = -RT ln( sum_i exp(-G_i / RT) )
    """
    print("\n" + SEP)
    print("  STEP 5 -- Boltzmann aggregation")
    print(SEP)
    print(f"  INPUT : {len(thermo_results)} molecule(s), T={temperature} K")

    RT    = _R * temperature / (_cal * 1000)
    g_eff = {}

    for smi, thermo_list in thermo_results.items():
        if not thermo_list:
            print(f"  WARNING: no thermo data for {smi!r}, skipping.")
            continue

        Gs    = np.array([t["G"] for t in thermo_list])
        G_min = Gs.min()
        log_Z = np.log(np.sum(np.exp(-(Gs - G_min) / RT)))
        G_boltz = G_min - RT * log_Z
        weights = np.exp(-(Gs - G_min) / RT) / np.exp(log_Z)

        print(f"\n  [{smi}]")
        for i, (g, w) in enumerate(zip(Gs, weights)):
            print(f"    Conformer {i}: G = {g:.4f} kcal/mol   weight = {w:.4f}")
        print(f"    G_eff = {G_boltz:.4f} kcal/mol")

        g_eff[smi] = G_boltz

    print(f"\n  OUTPUT: G_eff for {len(g_eff)} molecule(s)")
    return g_eff


# =============================================================================
#  STEP 6 -- dG profile per reaction
# =============================================================================

def step6_compute_profile(reactions, g_eff, mol_failures=None):
    """
    Input : reactions, g_eff {smiles: G_eff}, mol_failures {smiles: reason}
    Output: (profiles dict, rxn_failures dict {rxn_id: reason})
    """
    print("\n" + SEP)
    print("  STEP 6 -- Compute dG profiles")
    print(SEP)
    print(f"  INPUT : {len(reactions)} reaction(s), {len(g_eff)} molecule G values")

    profiles    = {}
    rxn_failures = {}
    mol_failures = mol_failures or {}

    for rxn in reactions:
        rxn_id = rxn["rxn_id"]
        states = rxn["states"]
        print(f"\n  [{rxn_id}]")

        state_G        = []
        failed_mols    = []
        for state in states:
            mols    = state.get("smiles", state["raw_smiles"])
            missing = [m for m in mols if m not in g_eff]
            if missing:
                failed_reasons = [
                    f"{m}: {mol_failures.get(m, 'G not computed')}"
                    for m in missing
                ]
                print(f"    WARNING: missing G for {missing} in {state['label']}")
                state_G.append(None)
                failed_mols.extend(failed_reasons)
            else:
                state_G.append(sum(g_eff[m] for m in mols))

        valid_Gs = [g for g in state_G if g is not None]
        if not valid_Gs:
            reason = f"No G values computed. Failed molecules: {'; '.join(failed_mols)}"
            print(f"    ERROR: no G values for {rxn_id} -- skipping.")
            rxn_failures[rxn_id] = reason
            continue

        if state_G[0] is None:
            reason = (f"State_0 (reference) could not be computed. "
                      f"Failed molecules: {'; '.join(failed_mols)}")
            print(f"    ERROR: State_0 (reference) is missing G values for {rxn_id} -- skipping.")
            print(f"           The reference state must be computable. Fix the reactant SMILES.")
            rxn_failures[rxn_id] = reason
            continue

        G0      = state_G[0]
        profile = []
        prev_dg = 0.0
        for state, G_abs in zip(states, state_G):
            mols = state.get("smiles", state["raw_smiles"])
            dG   = (G_abs - G0) if G_abs is not None else None
            ddG  = (dG - prev_dg) if dG is not None else None
            profile.append({
                "label":     state["label"],
                "molecules": mols,
                "G_abs":     G_abs,
                "dG":        dG,
                "ddG":       ddG,
            })
            if dG is not None:
                sign = "+" if ddG > 0 else ""
                print(f"    {state['label']:12s}  dG = {dG:+8.2f}  "
                      f"ddG = {sign}{ddG:.2f}  ({' + '.join(mols)})")
                prev_dg = dG

        profiles[rxn_id] = profile

    print(f"\n  OUTPUT: profiles for {len(profiles)} reaction(s)"
          f"  |  {len(rxn_failures)} failure(s)")
    return profiles, rxn_failures


# =============================================================================
#  STEP 7 -- Staircase plot
# =============================================================================

def step7_plot(profiles, unit="kcal/mol", step_width=0.6, step_gap=0.5,
               output_dir="."):
    """
    Input : profiles from Step 6
    Output: one PNG per reaction, saved as {output_dir}/{rxn_id}.png

    If `profiles` is empty (e.g. every reaction failed upstream), this
    function prints a clear error and returns without calling matplotlib,
    instead of crashing inside plt.subplots with nrows=0.
    """
    print("\n" + SEP)
    print("  STEP 7 -- Staircase plot")
    print(SEP)
    print(f"  INPUT : {len(profiles)} profile(s)  ->  {output_dir}/{{rxn_id}}.png")

    if not profiles:
        print("  ERROR: no profiles to plot -- every reaction failed in an "
              "earlier step (see Step 4/5/6 output above for the root cause).")
        print("  Skipping plot generation.")
        return

    saved = []
    for rxn_id, profile in profiles.items():
        valid   = [s for s in profile if s["dG"] is not None]
        dGs     = [s["dG"]        for s in valid]
        labels  = [s["label"]     for s in valid]
        mols    = [s["molecules"] for s in valid]
        n       = len(dGs)

        if n == 0:
            print(f"  WARNING: {rxn_id} has no valid states to plot -- skipping.")
            continue

        fig, ax = plt.subplots(figsize=(max(8, n * 2), 5))

        pitch     = step_width + step_gap
        x_centers = np.arange(n) * pitch
        x_left    = x_centers - step_width / 2
        x_right   = x_centers + step_width / 2

        def col(i):
            if i == 0:
                return "#1f77b4"
            return "#d62728" if dGs[i] > dGs[i-1] else "#2ca02c"

        for i in range(n):
            c = col(i)
            ax.hlines(dGs[i], x_left[i], x_right[i], colors=c, linewidths=3.5, zorder=3)
            sign = "+" if dGs[i] > 0 else ""
            ax.text(x_centers[i], dGs[i] + 0.5, f"{sign}{dGs[i]:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=c)
            mol_str = " + ".join(mols[i])
            ax.text(x_centers[i], dGs[i] - 0.8,
                    f"{labels[i]}\n({mol_str})",
                    ha="center", va="top", fontsize=7.5, color="#444")
            if i < n - 1:
                ax.plot([x_right[i], x_left[i+1]], [dGs[i], dGs[i+1]],
                        color="#aaa", linewidth=1.2, linestyle="--", zorder=2)
            if i > 0:
                ddg   = dGs[i] - dGs[i-1]
                mid_x = (x_right[i-1] + x_left[i]) / 2
                ax.annotate("", xy=(mid_x, dGs[i]), xytext=(mid_x, dGs[i-1]),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
                sign2 = "+" if ddg > 0 else ""
                ax.text(mid_x + 0.04, (dGs[i] + dGs[i-1]) / 2,
                        f"{sign2}{ddg:.1f}", fontsize=7.5, color=c, va="center")

        y_pad = max(2.0, (max(dGs) - min(dGs)) * 0.2)
        ax.set_ylim(min(dGs) - y_pad * 2, max(dGs) + y_pad)
        ax.set_xlim(x_left[0] - step_gap, x_right[-1] + step_gap)
        ax.axhline(0, color="#ccc", linewidth=0.8, linestyle=":")
        ax.set_xticks([])
        ax.set_ylabel(f"dG ({unit})", fontsize=11)
        ax.set_title(f"Free Energy Profile -- {rxn_id}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right", "bottom"]].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{rxn_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"    saved: {out_path}")

    print(f"\n  OUTPUT: {len(saved)} plot(s) saved to {output_dir}/")


# =============================================================================
#  PARALLEL WORKER -- process one molecule through Steps 3 + 4
# =============================================================================

def _process_molecule(args):
    """
    Worker function for parallel execution.
    Runs Steps 3 and 4 for a single molecule and returns its thermo list.
    Designed to be called by ProcessPoolExecutor -- must be top-level and
    accept a single argument (pickling requirement).

    args : tuple of (smi, rdkit_mol, solvent, crest_binary, n_cores_crest, output_dir, save_geoms)

    Each worker creates its own temporary directory for Vibrations cache files
    so that parallel processes never write to the same path simultaneously.
    """
    smi, rdmol, solvent, crest_binary, n_cores_crest, output_dir, save_geoms = args

    with tempfile.TemporaryDirectory(prefix="vib_worker_") as vib_dir:

        # Step 3 for this molecule only
        single_ensemble = step3_crest_sampling(
            {smi: rdmol},
            solvent=solvent,
            n_cores=n_cores_crest,
            crest_binary=crest_binary,
        )

        # Step 4 for this molecule only -- vib files go into the worker tempdir
        single_thermo, single_failures = step4_xtb_optimize_and_thermo(
            single_ensemble,
            solvent=solvent,
            output_dir=output_dir,
            save_geoms=save_geoms,
            vib_dir=vib_dir,
        )

    thermo_list   = single_thermo.get(smi, [])
    worker_failures = single_failures if single_failures else {}
    return smi, thermo_list, worker_failures



# =============================================================================
#  FAILURE REPORT
# =============================================================================

def write_failure_report(rxn_failures, mol_failures, output_dir="."):
    """
    Write a plain-text failure report listing every reaction that could not
    produce a plot, with the specific molecule(s) and error reason(s).

    Output: {output_dir}/failed_reactions.txt
    """
    if not rxn_failures and not mol_failures:
        return

    report_path = os.path.join(output_dir, "failed_reactions.txt")
    lines = [
        "Simple Reaction Thermo -- Failure Report",
        "=" * 60,
        "",
    ]

    if rxn_failures:
        lines.append(f"REACTIONS WITHOUT PLOTS ({len(rxn_failures)})")
        lines.append("-" * 60)
        for rxn_id, reason in rxn_failures.items():
            lines.append(f"  {rxn_id}")
            lines.append(f"    Reason : {reason}")
            lines.append("")

    if mol_failures:
        lines.append(f"MOLECULES THAT FAILED IN xTB ({len(mol_failures)})")
        lines.append("-" * 60)
        for smi, reason in mol_failures.items():
            lines.append(f"  SMILES : {smi}")
            lines.append(f"  Reason : {reason}")
            lines.append("")

    lines.append("=" * 60)
    lines.append("Tip: check SMILES validity, charge/spin state, and whether")
    lines.append("xTB supports the elements in your molecule.")

    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Failure report written to: {report_path}")


# =============================================================================
#  MAIN
# =============================================================================

def run_pipeline(csv_path, solvent="water", crest_binary="crest",
                 parallel=False, n_workers=4, n_cores_crest=2,
                 output_dir=".", save_geoms=False):
    """
    Parameters
    ----------
    csv_path      : path to reactions CSV
    solvent       : solvent name for CREST/xTB, or 'none' for gas phase
    crest_binary  : path or name of the CREST executable
    parallel      : if True, run Steps 3+4 in parallel across molecules
    n_workers     : number of parallel processes (parallel=True only)
    n_cores_crest : CPU cores given to each CREST call; when parallel=True,
                    set so that n_workers * n_cores_crest <= total cores
    output_dir    : directory to save output files (plot and optimized conformers)
    save_geoms    : if True, save optimized conformer geometries in output_dir
    """
    if not ASE_AVAILABLE:
        print("\n" + SEP)
        print("  FATAL: ASE/tblite import failed at startup (see "
              "[IMPORT WARNING] above).")
        print("  Steps 4-7 cannot run without ASE/tblite. Check that you are "
              "using the correct Python interpreter/conda environment:")
        print("    which python")
        print("    python -c \"from tblite.ase import TBLite; print('OK')\"")
        print(SEP)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    reactions     = step0_parse_csv(csv_path)
    unique_smiles = step1_extract_unique_molecules(reactions)
    step1b_check_atom_balance(reactions)
    mol_3d        = step2_generate_3d(unique_smiles)

    if parallel:
        print("\n" + SEP)
        print("  STEPS 3+4 -- Parallel execution")
        print(SEP)
        print(f"  n_workers={n_workers}  n_cores_crest={n_cores_crest}")
        print(f"  total cores used <= {n_workers * n_cores_crest}")

        thermo       = {}
        mol_failures = {}
        args_list = [
            (smi, mol_3d[smi], solvent, crest_binary, n_cores_crest, output_dir, save_geoms)
            for smi in unique_smiles
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_molecule, args): args[0]
                for args in args_list
            }
            for future in as_completed(futures):
                smi = futures[future]
                try:
                    smi_out, thermo_list, worker_failures = future.result()
                    thermo[smi_out] = thermo_list
                    mol_failures.update(worker_failures)
                    print(f"  DONE: {smi}")
                except Exception as e:
                    reason = f"{type(e).__name__}: {e}"
                    print(f"  ERROR: {smi} -> {reason}")
                    thermo[smi] = []
                    mol_failures[smi] = reason
    else:
        ensembles = step3_crest_sampling(mol_3d, solvent=solvent,
                                          crest_binary=crest_binary,
                                          n_cores=n_cores_crest)
        thermo, mol_failures = step4_xtb_optimize_and_thermo(
            ensembles, solvent=solvent,
            output_dir=output_dir, save_geoms=save_geoms)

    g_eff             = step5_boltzmann_aggregate(thermo)
    profiles, rxn_failures = step6_compute_profile(reactions, g_eff, mol_failures)
    write_failure_report(rxn_failures, mol_failures, output_dir=output_dir or ".")

    step7_plot(profiles, output_dir=output_dir or ".")

    return profiles


if __name__ == "__main__":
    # Check if any argument starts with "-" to determine if we use flags or positional arguments
    use_flags = any(arg.startswith("-") for arg in sys.argv[1:])

    if use_flags or len(sys.argv) == 1:
        import argparse
        parser = argparse.ArgumentParser(description="Modular free energy profile pipeline for reaction pathways.")
        parser.add_argument("reactions_csv", help="Path to reactions CSV file")
        parser.add_argument("--solvent", default="water", help="Solvent name for CREST/xTB, or 'none' for gas phase (default: water)")
        parser.add_argument("--crest-binary", default="crest", help="Path or name of the CREST executable (default: crest)")
        parser.add_argument("--parallel", action="store_true", help="Run Steps 3+4 in parallel across molecules")
        parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel processes (default: 4)")
        parser.add_argument("--n-cores-crest", type=int, default=2, help="CPU cores given to each CREST call (default: 2)")
        parser.add_argument("--output-dir", default=".", help="Directory to save plots and optimized geometries (default: .)")
        parser.add_argument("--save-geoms", action="store_true", help="Save the optimized conformer geometries as XYZ files in the output directory")

        args = parser.parse_args()
        csv_path      = args.reactions_csv
        solvent       = args.solvent
        crest_binary  = args.crest_binary
        parallel      = args.parallel
        n_workers     = args.n_workers
        n_cores_crest = args.n_cores_crest
        output_dir    = args.output_dir
        save_geoms    = args.save_geoms
    else:
        csv_path      = sys.argv[1]
        solvent       = sys.argv[2] if len(sys.argv) > 2 else "water"
        crest_binary  = sys.argv[3] if len(sys.argv) > 3 else "crest"
        parallel      = sys.argv[4].lower() in ("true", "1", "yes") if len(sys.argv) > 4 else False
        n_workers     = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        n_cores_crest = int(sys.argv[6]) if len(sys.argv) > 6 else 2
        output_dir    = sys.argv[7] if len(sys.argv) > 7 else "."
        save_geoms    = sys.argv[8].lower() in ("true", "1", "yes") if len(sys.argv) > 8 else False

    run_pipeline(csv_path, solvent=solvent, crest_binary=crest_binary,
                 parallel=parallel, n_workers=n_workers,
                 n_cores_crest=n_cores_crest, output_dir=output_dir,
                 save_geoms=save_geoms)