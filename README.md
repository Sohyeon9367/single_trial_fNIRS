### single_trial_fNIRS

Single-trial fNIRS hemodynamic latency analysis pipeline for investigating neurovascular dysfunction across the Alzheimer’s disease (AD) continuum.

This repository contains MATLAB scripts required to reproduce single-trial latency modeling, linear mixed-effects analysis, topographic visualization, behavioral correlations, and Leave-One-Subject-Out (LOSO) classification.

---

## 📂 Repository Structure


single_trial_fNIRS/
│
├── A0_MBLL.m
├── A1_build_stroop_group_structure.m
├── A2_LMEandKDE.m
├── A3_averaged_beta_map.m
├── A4_Advanced_topographic_map.m
├── A5_behaviour_hemodynamics_correlation.m
├── A6_run_loso_full.m
│
├── build_subject_features_latency.m
│
└── Stroop_AllSubjects_NIRS_HRF_TTest_05.zip


---

## 🧠 Analysis Overview

The pipeline performs:

1. **MBLL conversion** (raw light intensity → HbO/HbR)
2. **Trial-level GLM modeling**
3. **Single-trial peak latency extraction**
4. **Linear Mixed-Effects modeling**


Latency ~ Group * ACC + Age + Education + (1|SubjectID)


5. **Kernel Density Estimation (KDE)** of latency distributions
6. **Leave-One-Subject-Out (LOSO) classification**
- Latency-based features
- Beta difference–based features
- Combined model
- Permutation-based AUC significance testing

---

## 📦 Data Availability

The preprocessed dataset (compressed, 65MB) is included in this repository:


Stroop_AllSubjects_NIRS_HRF_TTest_05.zip


All subjects are anonymized and identified using research codes (e.g., HC01, MCI03, AD07).  
No personally identifiable information is included.

---

## 🔁 Reproducing the Main Results

Step 1 — Unzip dataset

Extract:


Stroop_AllSubjects_NIRS_HRF_TTest_05.zip


Place the extracted `.mat` file in the repository root directory.

---

Step 2 — Run latency analysis

In MATLAB:

```matlab
A2_LMEandKDE

This generates:

Trial-level latency table

Linear mixed-effects (LME) results

KDE-based latency distributions

Step 3 — Topographic mapping
A3_averaged_beta_map
A4_Advanced_topographic_map

These scripts generate channel-wise averaged beta maps and advanced topographic visualizations.

Step 4 — Correlation with behavioral data
A5_behaviour_hemodynamics_correlation

Performs correlation analyses between hemodynamic features and behavioral performance metrics.

Step 5 — Run classification
A6_run_loso_full

This performs:

Pairwise HC vs MCI vs AD classification

LOSO logistic regression

Permutation-based AUC significance testing

🛠 Requirements

MATLAB (R2020b or later recommended)

Statistics and Machine Learning Toolbox

Signal Processing Toolbox

SPM (for spm_hrf) added to the MATLAB path

👤 Maintainer

Sohyeon Yoo
GitHub: https://github.com/Sohyeon9367
