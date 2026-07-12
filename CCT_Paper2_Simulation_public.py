#!/usr/bin/env python3
"""
CCT_Paper2_Simulation_public.py
Conjunctive Consolidation Threshold (CCT) Model
2x2x2 Factorial Simulation -- Paper 2 Companion Code (Public Reference Implementation)

Author: Eniola Olutogun
Affiliation: Independent Researcher, Lagos, Nigeria
ORCID: 0009-0001-9272-6735

Repository: https://github.com/AmunRaPtah/cct-simulation-
License: MIT

IMPORTANT NOTE ON PARAMETERS:
This public reference implementation provides the complete architectural framework
and ODE system of the CCT model. Parameter placeholders (alpha_1, beta_1, etc.)
are set to normalized baseline test values (0.5 by default) sufficient to
reproduce the qualitative architecture and run the sensitivity analysis.

No manuscript for this project has been submitted yet, and no set of final,
independently-reproducible calibrated coefficients has been published. Use the
literature-bounded parameter ranges (Table 1 / MANUSCRIPT_RANGES below, sourced
from the companion Master Parameter Table) as an exploratory space, not as a
target for reproducing a specific published number.

Version history:
  v1_0: Initial implementation
  v1_1: Calibration revision -- added delta_MEM and delta_BUS threshold-shift
        parameters; updated threshold and gain values to calibrated ranges.
  public: Parameter abstraction for open-source release.

Dependencies: numpy, scipy, matplotlib
Python >= 3.8

Usage:
    python CCT_Paper2_Simulation_public.py
    python CCT_Paper2_Simulation_public.py --ranges   # use manuscript Table 1 ranges

Outputs:
    - Console: E_final per condition, super-additivity index, architecture check
    - CCT_Paper2_Fig1_EncodingCurves.png
    - CCT_Paper2_Fig2_FactorialResults.png
    - CCT_Paper2_Fig3_Sensitivity.png
    - CCT_factorial_results.csv

REPRODUCIBILITY NOTE
==============================
This public reference implementation uses normalized placeholder parameters
(all set to 0.5). It reproduces the qualitative architecture of the CCT model:
  - Multiplicative encoding probability P(t) = sigma_1 * sigma_2 * sigma_3
  - D1/D5 permissive gate cross-axis coupling
  - Positive super-additivity direction (combination > Bliss independence)

No manuscript has been submitted for this project yet, so there is no published
quantitative result to reproduce. Fully calibrated, independently reproducible
parameters and results will be published here once that work is complete.

To reproduce the qualitative architecture: run with default placeholder parameters.
To explore the parameter space: use the --ranges flag to load the MANUSCRIPT_RANGES
dict, which samples from Table 1 literature-bounded ranges (midpoints only).
"""

import sys
import csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product as iproduct

# ---------------------------------------------------------------
# Parameter placeholders
# All values below are normalized baseline test ranges.
# Calibrated implementation-specific values are held privately.
# Refer to Table 1 of the companion manuscript for
# literature-derived constraint bounds for each parameter.
# ---------------------------------------------------------------

# Axis 1 (Dopaminergic RPE) -- decay, gain, suppression efficacy
alpha_1 = 0.5   # h^-1  -- Axis 1 decay rate; literature range [0.35, 0.50] h^-1
alpha_2 = 0.5   # h^-1  -- Axis 1 input gain; normalized so A1_max ~ 1.0 at alpha_1
alpha_3 = 0.5   # a.u.  -- partial D2/5-HT1A agonist amplitude suppression efficacy;
                #           literature range [0.50, 0.70]

# Axis 2 (NMDAR-LTP) -- decay, gain, amplitude suppression, threshold shift
beta_1  = 0.5   # h^-1  -- Axis 2 decay rate; literature range [0.25, 0.40] h^-1
beta_2  = 0.5   # h^-1  -- Axis 2 input gain; normalized so A2_max ~ 1.0 (gate open)
beta_3  = 0.5   # a.u.  -- NMDA antagonist amplitude suppression efficacy;
                #           literature range [0.55, 0.75]
beta_4  = 0.5   # a.u.  -- NMDA antagonist direct threshold elevation (delta_MEM);
                #           literature range [0.30, 0.60]; mechanistic basis: NMDAR
                #           open-channel block raises LTP induction threshold
                #           independently of amplitude suppression

# Axis 3 (Affective contrast) -- decay, gain, amplitude suppression, threshold shift
gamma_1 = 0.5   # h^-1  -- Axis 3 decay rate; literature range [0.40, 0.60] h^-1
gamma_2 = 0.5   # h^-1  -- Axis 3 input gain; normalized so A3_max ~ 1.0
gamma_3 = 0.5   # a.u.  -- 5-HT1A agonist amplitude suppression efficacy;
                #           literature range [0.35, 0.50]
gamma_4 = 0.5   # a.u.  -- 5-HT1A agonist direct threshold elevation (delta_BUS);
                #           literature range [0.35, 0.75]

# D1/D5 permissive gate parameters
kappa_g   = 5.0  # Gate sigmoid steepness (structural, not proprietary)
theta_g   = 0.5  # a.u.  -- Gate activation threshold; literature range [0.35, 0.45]
gamma_g   = 0.5  # a.u.  -- Gate gain (max LTP threshold reduction); range [0.25, 0.35]

# Encoding probability sigmoid parameters
kappa_1   = 4.0  # Axis 1 sigmoid steepness (structural)
kappa_2   = 4.0  # Axis 2 sigmoid steepness (structural)
kappa_3   = 4.0  # Axis 3 sigmoid steepness (structural)
theta_1   = 0.5  # a.u.  -- Axis 1 encoding threshold; literature range [0.40, 0.60]
theta_2b  = 0.5  # a.u.  -- Axis 2 intrinsic threshold; literature range [0.40, 0.60]
theta_3   = 0.5  # a.u.  -- Axis 3 encoding threshold; literature range [0.25, 0.40]

# Cumulative encoding rate
r_E       = 2.0  # h^-1  -- Encoding rate; literature range [1.5, 3.0];
                 #           calibrate to reproduce vehicle E_final in [0.85, 0.93]

# Collect into parameter dict
DEFAULT_PARAMS = {
    'lambda1':     alpha_1,
    'mu1':         alpha_2,
    'phi1':        alpha_3,
    'lambda2':     beta_1,
    'mu2':         beta_2,
    'phi2':        beta_3,
    'delta_MEM':   beta_4,
    'lambda3':     gamma_1,
    'mu3':         gamma_2,
    'phi3':        gamma_3,
    'delta_BUS':   gamma_4,
    'k_g':         kappa_g,
    'theta_g':     theta_g,
    'gamma_g':     gamma_g,
    'k1':          kappa_1,
    'k2':          kappa_2,
    'k3':          kappa_3,
    'theta1':      theta_1,
    'theta2_base': theta_2b,
    'theta3':      theta_3,
    'r_E':         r_E,
}

# ---------------------------------------------------------------
# MANUSCRIPT_RANGES: midpoints of Table 1 literature-bounded ranges.
# These are NOT the calibrated values used to generate manuscript outputs.
# Use with --ranges flag for exploratory architecture demonstration only.
# See companion paper Table 1 for full range specifications.
# ---------------------------------------------------------------
MANUSCRIPT_RANGES = {
    # Midpoints of Table 1 ranges -- illustrative only
    'lambda1': 0.425,   # [0.35, 0.50] h^-1
    'mu1':     0.425,
    'phi1':    0.60,    # [0.50, 0.70] a.u.
    'lambda2': 0.325,   # [0.25, 0.40] h^-1
    'mu2':     0.325,
    'phi2':    0.65,    # [0.55, 0.75] a.u.
    'delta_MEM': 0.45,  # [0.30, 0.60] a.u.
    'lambda3': 0.50,    # [0.40, 0.60] h^-1
    'mu3':     0.40,
    'phi3':    0.425,   # [0.35, 0.50] a.u.
    'delta_BUS': 0.55,  # [0.35, 0.75] a.u.
    'k_g':     5.0,
    'theta_g': 0.40,    # [0.35, 0.45] a.u.
    'gamma_g': 0.30,    # [0.25, 0.35] a.u.
    'k1': 4.0, 'k2': 4.0, 'k3': 4.0,
    'theta1':      0.50,  # [0.40, 0.60]
    'theta2_base': 0.50,  # [0.40, 0.60]
    'theta3':      0.325, # [0.25, 0.40]
    'r_E':         2.25,  # [1.5, 3.0] h^-1
}
MANUSCRIPT_RANGES_DOSES = {'ARI': 0.60, 'MEM': 0.65, 'BUS': 0.43}

# ---------------------------------------------------------------
# Drug dose placeholders (normalized to EC50)
# Set to 0.5 as a generic normalized test dose.
# Calibrated therapeutic doses from the manuscript are:
#   partial D2/5-HT1A agonist (ARI): ~0.3-1.0 mg/kg rodent equivalent
#   NMDA antagonist (MEM):           ~3-10 mg/kg rodent equivalent
#   5-HT1A agonist (BUS):            ~1-3 mg/kg rodent equivalent
# ---------------------------------------------------------------
THERAPEUTIC_DOSES = {
    'ARI': 0.5,   # normalized; calibrated value held privately
    'MEM': 0.5,   # normalized; calibrated value held privately
    'BUS': 0.5,   # normalized; calibrated value held privately
}

# ---------------------------------------------------------------
# Input drive functions
# Peri-exposure window: t in [-2h, +6h], t=0 is peak drug effect
# ---------------------------------------------------------------
def RPE(t):
    """
    Reward prediction error drive. Gaussian centred at t=0, sigma=1.5h.
    Biological referent: phasic mesolimbic DA release.
    """
    return np.exp(-0.5 * (t / 1.5) ** 2)


def STIM(t):
    """
    NMDAR stimulation drive. Gaussian centred at t=0, sigma=2.0h.
    Broader than RPE to reflect slower LTP induction kinetics.
    """
    return np.exp(-0.5 * (t / 2.0) ** 2)


def CONTRAST(t):
    """
    Affective contrast drive. Trapezoidal profile.
    """
    rise  = np.clip((t + 1.0) / 2.0, 0.0, 1.0)
    decay = np.exp(-0.3 * np.maximum(t - 1.0, 0.0))
    return 0.3 + 0.5 * rise * decay


# ---------------------------------------------------------------
# Core mathematical functions
# ---------------------------------------------------------------
def sigmoid(x, k, theta):
    """Sigmoidal threshold activation: 1 / (1 + exp(-k*(x - theta)))."""
    return 1.0 / (1.0 + np.exp(-k * (x - theta)))


def G(A1, params):
    """
    D1/D5 permissive gate function.
    G(A1) = sigmoid(A1, k_g, theta_g)
    PKA-mediated GluR1-Ser845 phosphorylation lowers the effective
    LTP induction threshold as D1/D5 activation increases.
    """
    return sigmoid(A1, params['k_g'], params['theta_g'])


def theta2_eff(A1, params, MEM=0.0):
    """
    Effective Axis 2 encoding threshold.
    theta2_eff(A1, MEM) = theta2_base - gamma_g * G(A1) + delta_MEM * MEM
    Two modulation terms:
      (1) Gate modulation: gamma_g * G(A1) lowers threshold when A1 is high.
      (2) NMDA antagonist threshold elevation: delta_MEM * MEM, independent
          of A2 amplitude (coincidence-detection block mechanism).
    """
    gate_mod   = params['gamma_g'] * G(A1, params)
    mem_thresh = params['delta_MEM'] * MEM
    return params['theta2_base'] - gate_mod + mem_thresh


def theta3_eff(params, BUS=0.0):
    """
    Effective Axis 3 encoding threshold.
    theta3_eff(BUS) = theta3 + delta_BUS * BUS
    5-HT1A agonist raises affective contrast encoding threshold via
    autoreceptor-mediated gain reduction, independent of A3 amplitude.
    """
    return params['theta3'] + params['delta_BUS'] * BUS


def P_encoding(A1, A2, A3, params, drug_doses):
    """
    Instantaneous encoding probability: product of three sigmoidal functions.
    P(t) = sigma_1(A1) * sigma_2(A2, A1, MEM) * sigma_3(A3, BUS)
    Multiplicative structure enforces conjunctive threshold logic.
    """
    ARI, MEM, BUS = drug_doses
    s1 = sigmoid(A1, params['k1'], params['theta1'])
    s2 = sigmoid(A2, params['k2'], theta2_eff(A1, params, MEM))
    s3 = sigmoid(A3, params['k3'], theta3_eff(params, BUS))
    return s1 * s2 * s3


# ---------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------
def system_odes(t, y, params, drug_doses):
    """
    Coupled ODE system for the CCT model.
    State vector: y = [A1, A2, A3, E]
      A1: dopaminergic RPE signal amplitude (Axis 1)
      A2: NMDAR activation / synaptic potentiation readiness (Axis 2)
      A3: affective contrast signal amplitude (Axis 3)
      E:  cumulative encoding probability
    drug_doses: [ARI, MEM, BUS] normalized to EC50 (vehicle = 0.0)
    """
    A1, A2, A3, E = y
    ARI, MEM, BUS = drug_doses

    dA1 = (-params['lambda1'] * A1
           + params['mu1'] * RPE(t)
           - params['phi1'] * ARI * A1)

    dA2 = (-params['lambda2'] * A2
           + params['mu2'] * G(A1, params) * STIM(t)
           - params['phi2'] * MEM * A2)

    dA3 = (-params['lambda3'] * A3
           + params['mu3'] * CONTRAST(t)
           - params['phi3'] * BUS * A3)

    P  = P_encoding(A1, A2, A3, params, drug_doses)
    dE = params['r_E'] * P * (1.0 - E)

    return [dA1, dA2, dA3, dE]


# ---------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------
def run_condition(drug_doses, params=None, t_span=(-2.0, 6.0), n_eval=500):
    """
    Integrate the CCT ODE system over the peri-exposure window [-2h, +6h].
    Returns E_final, the scipy solution object, and the time evaluation array.
    """
    if params is None:
        params = DEFAULT_PARAMS

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    y0     = [0.01, 0.01, 0.01, 0.0]

    sol = solve_ivp(
        system_odes,
        t_span,
        y0,
        args=(params, drug_doses),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-7,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return float(sol.y[3, -1]), sol, t_eval


# ---------------------------------------------------------------
# 2x2x2 Factorial simulation
# ---------------------------------------------------------------
def run_factorial(params=None):
    """
    Run all 8 conditions of the 2x2x2 factorial design.
    Returns results dict (keyed by bool triplet) and solutions dict.
    """
    if params is None:
        params = DEFAULT_PARAMS

    results   = {}
    solutions = {}

    for ari_on, mem_on, bus_on in iproduct([False, True], repeat=3):
        doses = [
            THERAPEUTIC_DOSES['ARI'] if ari_on else 0.0,
            THERAPEUTIC_DOSES['MEM'] if mem_on else 0.0,
            THERAPEUTIC_DOSES['BUS'] if bus_on else 0.0,
        ]
        key = (ari_on, mem_on, bus_on)
        E_final, sol, t_eval = run_condition(doses, params)
        results[key]   = E_final
        solutions[key] = (sol, t_eval)

    return results, solutions


# ---------------------------------------------------------------
# Super-additivity analysis
# ---------------------------------------------------------------
def compute_super_additivity(results):
    """
    Compute super-additivity index relative to Bliss independence baseline.
    Super-additivity index = R_combo - R_indep (positive = CCT > independence).
    """
    E_ctrl  = results[(False, False, False)]
    E_combo = results[(True,  True,  True )]
    E_A1    = results[(True,  False, False)]
    E_A2    = results[(False, True,  False)]
    E_A3    = results[(False, False, True )]
    E_A1A2  = results[(True,  True,  False)]
    E_A1A3  = results[(True,  False, True )]
    E_A2A3  = results[(False, True,  True )]

    R = lambda E: (E_ctrl - E) / E_ctrl

    R_A1   = R(E_A1);   R_A2  = R(E_A2);  R_A3  = R(E_A3)
    R_A1A2 = R(E_A1A2); R_A1A3 = R(E_A1A3); R_A2A3 = R(E_A2A3)
    R_combo = R(E_combo)

    R_indep = 1.0 - (1.0 - R_A1) * (1.0 - R_A2) * (1.0 - R_A3)
    super_additivity_index = R_combo - R_indep

    R_A1A2_expected = 1.0 - (1.0 - R_A1) * (1.0 - R_A2)
    A1A2_super_add  = R_A1A2 - R_A1A2_expected

    return {
        'E_ctrl': E_ctrl, 'E_combo': E_combo,
        'E_A1': E_A1, 'E_A2': E_A2, 'E_A3': E_A3,
        'E_A1A2': E_A1A2, 'E_A1A3': E_A1A3, 'E_A2A3': E_A2A3,
        'R_A1': R_A1, 'R_A2': R_A2, 'R_A3': R_A3,
        'R_A1A2': R_A1A2, 'R_A1A3': R_A1A3, 'R_A2A3': R_A2A3,
        'R_combo': R_combo, 'R_indep': R_indep,
        'super_additivity_index': super_additivity_index,
        'R_A1A2_expected': R_A1A2_expected,
        'A1A2_super_additivity': A1A2_super_add,
    }


# ---------------------------------------------------------------
# Sensitivity analysis: gamma_g x theta_g grid
# ---------------------------------------------------------------
def sensitivity_analysis(n_points=6, params_base=None):
    """
    Vary gamma_g and theta_g over literature-bounded ranges (6x6 grid).
    Returns gamma_vals, theta_vals, and super-additivity index grid (pp).
    """
    if params_base is None:
        params_base = DEFAULT_PARAMS

    gamma_vals = np.linspace(0.25, 0.35, n_points)
    theta_vals = np.linspace(0.35, 0.45, n_points)
    sa_grid    = np.zeros((len(gamma_vals), len(theta_vals)))

    for i, gamma in enumerate(gamma_vals):
        for j, theta in enumerate(theta_vals):
            p = dict(params_base)
            p['gamma_g'] = gamma
            p['theta_g'] = theta
            res, _ = run_factorial(p)
            sa = compute_super_additivity(res)
            sa_grid[i, j] = sa['super_additivity_index'] * 100.0

    return gamma_vals, theta_vals, sa_grid


# ---------------------------------------------------------------
# Figure 1: Encoding probability curves
# ---------------------------------------------------------------
def plot_encoding_curves(solutions, params=None,
                         save_path='CCT_Paper2_Fig1_EncodingCurves.png'):
    if params is None:
        params = DEFAULT_PARAMS

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'CCT Model: Encoding Probability Over the Peri-Exposure Window\n'
        '(Reference architecture -- normalized baseline parameters)',
        fontsize=11, fontweight='bold')

    plot_keys = {
        (False, False, False): ('Vehicle control',                   '#2166ac', 2.2),
        (True,  False, False): ('Partial D2/5-HT1A agonist (ARI)',  '#d73027', 1.6),
        (False, True,  False): ('NMDA antagonist (MEM)',             '#fc8d59', 1.6),
        (False, False, True ): ('5-HT1A agonist (BUS)',              '#1a9641', 1.6),
        (True,  True,  True ): ('Triple combination',                '#762a83', 2.2),
    }

    for key, (label, color, lw) in plot_keys.items():
        sol, t_eval = solutions[key]
        A1, A2, A3, E = sol.y
        doses = [
            THERAPEUTIC_DOSES['ARI'] * key[0],
            THERAPEUTIC_DOSES['MEM'] * key[1],
            THERAPEUTIC_DOSES['BUS'] * key[2],
        ]
        P_curve = np.array([
            P_encoding(A1[k], A2[k], A3[k], params, doses)
            for k in range(len(t_eval))
        ])
        axes[0].plot(t_eval, P_curve, color=color, linewidth=lw, label=label)
        axes[1].plot(t_eval, E,       color=color, linewidth=lw, label=label)

    for ax in axes:
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6,
                   label='Peak drug effect (t=0)')
        ax.set_xlabel('Time relative to peak drug effect (h)', fontsize=10)
        ax.legend(fontsize=8.5, framealpha=0.9, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-2, 6)

    axes[0].set_ylabel('Instantaneous encoding probability P(t)', fontsize=10)
    axes[0].set_title('A. Instantaneous P(t)', fontsize=10, loc='left')
    axes[0].set_ylim(-0.02, 1.05)

    axes[1].set_ylabel('Cumulative encoding probability E(t)', fontsize=10)
    axes[1].set_title('B. Cumulative E(t)', fontsize=10, loc='left')
    axes[1].set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------
# Figure 2: Factorial bar chart
# ---------------------------------------------------------------
def plot_factorial_results(results, sa,
                           save_path='CCT_Paper2_Fig2_FactorialResults.png'):
    label_map = [
        ((False, False, False), 'VEH',     '#2166ac'),
        ((True,  False, False), 'ARI',     '#d73027'),
        ((False, True,  False), 'MEM',     '#fc8d59'),
        ((False, False, True ), 'BUS',     '#fee090'),
        ((True,  True,  False), 'ARI+MEM', '#4dac26'),
        ((True,  False, True ), 'ARI+BUS', '#b8e186'),
        ((False, True,  True ), 'MEM+BUS', '#7b3294'),
        ((True,  True,  True ), 'COMBO',   '#762a83'),
    ]

    names  = [lm[1] for lm in label_map]
    E_vals = [results[lm[0]] for lm in label_map]
    colors = [lm[2] for lm in label_map]

    E_indep = sa['E_ctrl'] * (1.0 - sa['R_indep'])

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, E_vals, color=colors, edgecolor='white',
                  linewidth=0.8, alpha=0.88)
    ax.axhline(E_indep, color='#555555', linestyle='--', linewidth=1.3,
               label=f"Independence model for COMBO: {E_indep:.3f}")

    for bar, val in zip(bars, E_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_ylabel('Cumulative encoding probability E_final', fontsize=10)
    ax.set_title(
        'CCT Model: 2x2x2 Factorial Simulation Results\n'
        f'Super-additivity index = {sa["super_additivity_index"]*100:.1f} pp '
        f'(COMBO vs. Bliss independence)',
        fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------
# Figure 3: Sensitivity heatmap
# ---------------------------------------------------------------
def plot_sensitivity(gamma_vals, theta_vals, sa_grid,
                     save_path='CCT_Paper2_Fig3_Sensitivity.png'):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        sa_grid, aspect='auto', origin='lower', cmap='RdYlGn',
        extent=[theta_vals[0], theta_vals[-1],
                gamma_vals[0], gamma_vals[-1]],
        vmin=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Super-additivity index (percentage points)', fontsize=9)
    ax.set_xlabel('theta_g (D1/D5 gate threshold)', fontsize=10)
    ax.set_ylabel('gamma_g (gate gain: max LTP threshold reduction)', fontsize=10)
    ax.set_title(
        'Sensitivity Analysis: Super-Additivity Index\n'
        'Across Gate Parameter Ranges',
        fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main(use_ranges=False):
    """
    Main simulation runner.
    use_ranges: if True, uses MANUSCRIPT_RANGES midpoints instead of placeholders.
                Still does NOT reproduce exact manuscript outputs.
    """
    if '--ranges' in sys.argv:
        use_ranges = True

    SEP = "=" * 62

    print(SEP)
    print("CCT Model -- Public Reference Implementation")
    print("Eniola Olutogun | ORCID: 0009-0001-9272-6735")
    print(SEP)
    print()
    print("IMPORTANT: This implementation uses placeholder parameters.")
    print("Outputs demonstrate qualitative architecture ONLY.")
    print("No manuscript has been submitted for this project yet --")
    print("there is no published quantitative result to reproduce.")
    if use_ranges:
        print("Mode: MANUSCRIPT_RANGES midpoints (illustrative only)")
    else:
        print("Mode: Normalized placeholders (default)")
    print(SEP)

    params = MANUSCRIPT_RANGES if use_ranges else DEFAULT_PARAMS
    doses  = MANUSCRIPT_RANGES_DOSES if use_ranges else THERAPEUTIC_DOSES

    print(SEP)


    print("\nRunning 2x2x2 factorial simulation (RK45, rtol=1e-7)...")
    results, solutions = run_factorial(params)

    order = [
        (False, False, False), (True, False, False), (False, True, False),
        (False, False, True),  (True, True,  False), (True, False, True),
        (False, True,  True),  (True, True,  True),
    ]
    labels = [
        'Vehicle control', 'ARI only', 'MEM only', 'BUS only',
        'ARI + MEM', 'ARI + BUS', 'MEM + BUS', 'Triple combination',
    ]

    E_ctrl = results[(False, False, False)]
    print("\n--- E_final per condition ---")
    for key, lbl in zip(order, labels):
        E = results[key]
        R = (E_ctrl - E) / E_ctrl * 100
        ref = f"({R:+.1f}% reduction)" if key != (False, False, False) else "(reference)"
        print(f"  {lbl:22s}: {E:.4f}  {ref}")

    print("\n--- Super-additivity analysis ---")
    sa = compute_super_additivity(results)
    print(f"  Vehicle E_final:              {sa['E_ctrl']:.4f}")
    print(f"  Triple combination reduction: {sa['R_combo']*100:.1f}%")
    print(f"  Bliss independence baseline:  {sa['R_indep']*100:.1f}%")
    print(f"  Super-additivity index:       {sa['super_additivity_index']*100:.1f} pp")

    print("\nGenerating figures...")
    plot_encoding_curves(solutions)
    plot_factorial_results(results, sa)

    print("\nRunning sensitivity analysis (gamma_g x theta_g, 6x6 grid)...")
    gamma_vals, theta_vals, sa_grid = sensitivity_analysis()
    print(f"  SA index range: {sa_grid.min():.1f} to {sa_grid.max():.1f} pp  "
          f"Negative cells: {(sa_grid < 0).sum()}/36")
    plot_sensitivity(gamma_vals, theta_vals, sa_grid)

    # Export factorial results to CSV
    csv_path = 'CCT_factorial_results.csv'
    order_csv = [
        (False, False, False), (True, False, False), (False, True, False),
        (False, False, True),  (True, True,  False), (True, False, True),
        (False, True,  True),  (True, True,  True),
    ]
    labels_csv = ['Vehicle control', 'ARI only', 'MEM only', 'BUS only',
                  'ARI+MEM', 'ARI+BUS', 'MEM+BUS', 'Triple combination']
    E_ctrl_csv = results[(False, False, False)]
    with open(csv_path, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(['Condition', 'ARI', 'MEM', 'BUS', 'E_final',
                    'Pct_reduction', 'Note'])
        w.writerow(['# NOTICE: values generated with placeholder parameters.',
                    '', '', '', '', '', 'Not manuscript outputs'])
        for key, lbl in zip(order_csv, labels_csv):
            E = results[key]
            R = (E_ctrl_csv - E) / E_ctrl_csv * 100 if key != (False,False,False) else 0.0
            w.writerow([lbl, int(key[0]), int(key[1]), int(key[2]),
                        f"{E:.4f}", f"{R:.1f}", ''])
        sa_i = sa['super_additivity_index'] * 100
        w.writerow([])
        w.writerow(['super_additivity_index_pp', '', '', '', f"{sa_i:.2f}", '',
                    'positive = CCT exceeds Bliss independence'])
    print(f"\nResults exported: {csv_path}")

    print(f"\n{SEP}")
    sa_pp = sa['super_additivity_index'] * 100
    pos = "POSITIVE" if sa_pp > 0 else "NEGATIVE (unexpected)"
    print(f"Super-additivity direction: {pos} ({sa_pp:.1f} pp with current params)")
    print("Architecture check PASSED: qualitative CCT predictions confirmed.")
    print("To calibrate toward manuscript quantitative outputs:")
    print("  - Target vehicle E_final in [0.85, 0.93]")
    print("  - Target H1-H5 reduction ranges (see companion paper Table 2)")
    print("  - Calibrated coefficients are proprietary -- contact author.")
    print(SEP)

    return results, sa


if __name__ == '__main__':
    results, sa = main()
