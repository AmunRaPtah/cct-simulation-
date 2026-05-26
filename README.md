# CCT Model: Conjunctive Consolidation Threshold

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![OSF](https://img.shields.io/badge/OSF-preregistered-blue)

**Public reference implementation of the Conjunctive Consolidation Threshold (CCT) Model**

Companion code to the CCT Model Trilogy:

- **Paper 1 (Review):** Olutogun E (2026). The Conjunctive Consolidation Threshold model: a tripartite pharmacological framework for encoding prevention in reward memory. *Neuroscience and Biobehavioral Reviews* (in revision).
- **Paper 2 (Specification):** Olutogun E (2026). Formal Mathematical Specification of the CCT Model: Encoding Probability Dynamics and Quantitative Predictions for Preclinical Validation. Preprint, OSF.
- **Paper 3 (Human Translation):** Olutogun E (2026). Consolidation Threshold Dynamics in Human Drug-Associated Learning: Bayesian Parameter Integration, Population Heterogeneity, and Clinical Trial Architecture for the CCT Model. Preprint, OSF.

**Author:** Eniola Olutogun | OIQB, Lagos, Nigeria | ORCID: 0009-0001-9272-6735

---

## IMPORTANT: Intellectual Property Notice

> The core multi-axis coupling architecture, qualitative conjunctive threshold logic, D1/D5 permissive gate mechanism, and super-additivity prediction framework are disclosed herein as **defensive publication** to establish global prior art. Specific calibrated parameter sets, proprietary optimization workflows, full commercial implementation details, and exact quantitative tuning coefficients are held confidential and are the subject of pending provisional patent applications by the author (ORCID: 0009-0001-9272-6735). **Unauthorized commercial use is prohibited.**
>
> Quantitative outputs (E_final values, super-additivity percentages) reported in the manuscript were generated with proprietary calibrated parameters that are not disclosed herein. The public implementation uses placeholder parameters and reproduces only the qualitative architecture.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/AmunRaPtah/cct-simulation-.git
cd cct-simulation-

# Install dependencies
pip install -r requirements.txt

# Run with default placeholder parameters (qualitative architecture demo)
python CCT_Paper2_Simulation_public.py

# Run with Table 1 literature range midpoints (illustrative only)
python CCT_Paper2_Simulation_public.py --ranges
```

**Outputs generated:**
- `CCT_Paper2_Fig1_EncodingCurves.png` — encoding probability curves over peri-exposure window
- `CCT_Paper2_Fig2_FactorialResults.png` — 2x2x2 factorial bar chart
- `CCT_Paper2_Fig3_Sensitivity.png` — sensitivity heatmap (gamma_g x theta_g)
- `CCT_factorial_results.csv` — numerical results table

---

## What This Code Does (and Does Not Do)

| What it reproduces | Status |
|---|---|
| ODE system architecture (3-axis, permissive gate) | Yes |
| Multiplicative encoding probability P(t) | Yes |
| D1/D5 permissive gate cross-axis coupling | Yes |
| Positive super-additivity direction | Yes |
| Exact manuscript E_final values (e.g., vehicle 0.855) | No |
| Exact super-additivity index (12.8 pp) | No |
| Proprietary calibrated coefficients | No |

The `--ranges` flag loads Table 1 literature midpoints (not calibrated posteriors) and still does not reproduce manuscript outputs.

---

## File Manifest

| File | Description |
|---|---|
| `CCT_Paper2_Simulation_public.py` | Main simulation script: ODE system, 2x2x2 factorial, sensitivity analysis, figure generation |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |
| `.gitignore` | Python standard gitignore |
| `docs/CCT_Master_Parameter_Table_S1.md` | Master parameter table S1 (literature-derived ranges, all three papers) |
| `docs/CCT_Review_Paper1.md` | Paper 1: Theoretical and translational review (preprint) |
| `docs/CCT_Mathematical_Specification_Paper2.md` | Paper 2: Formal mathematical specification (preprint) |
| `docs/CCT_Human_Translation_Paper3.md` | Paper 3: Human translation and clinical trial design (preprint) |

---

## Model Overview

The CCT model proposes that drug-memory encoding requires **concurrent above-threshold activation** across three independent neural axes within an 8-hour peri-exposure window:

- **Axis 1:** Dopaminergic reward prediction error (RPE) amplitude
- **Axis 2:** NMDA receptor-dependent synaptic potentiation readiness
- **Axis 3:** Affective contrast signal amplitude

A **D1/D5 permissive gate** creates a cross-axis dependency: D1/D5 activation lowers the LTP induction threshold on Axis 2, such that suppression of Axis 1 amplifies the encoding-preventive effect of Axis 2 modulation. This gating architecture is the mechanistic source of the model's primary prediction: **simultaneous sub-threshold suppression of all three axes produces super-additive (non-linear) reduction in encoding probability**, exceeding the Bliss independence prediction by 9.9-16.1 percentage points across the sensitivity analysis (direction positive in all 36 tested gate parameter combinations).

The three proof-of-concept agents used to validate the architecture:

| Agent | Target | Axis |
|---|---|---|
| Aripiprazole (ARI) | Partial D2/5-HT1A agonist | 1 |
| Memantine (MEM) | NMDA receptor antagonist | 2 |
| Buspirone (BUS) | Selective 5-HT1A agonist | 3 |

The framework generalizes to any pharmacological agents engaging the corresponding receptor mechanisms.

---

## Citation

If you use this code, please cite:

```
Olutogun E (2026). Formal Mathematical Specification of the Conjunctive Consolidation 
Threshold Model: Encoding Probability Dynamics and Quantitative Predictions for 
Preclinical Validation. Preprint, OSF. ORCID: 0009-0001-9272-6735.
```

---

## License

MIT License. See `LICENSE` file.

Copyright (c) 2026 Eniola Olutogun. Commercial use of the model architecture requires authorization from the author.
