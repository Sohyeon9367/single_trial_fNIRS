# single_trial_fNIRS

Single-trial fNIRS hemodynamic latency analysis pipeline for studying neurovascular dysfunction across the Alzheimer’s disease (AD) continuum.

This repository contains all MATLAB scripts required to reproduce the latency modeling, linear mixed-effects analysis, and LOSO classification results.

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

1. MBLL conversion (raw → HbO/HbR)
2. Trial-level GLM modeling
3. Single-trial peak latency extraction
4. Linear Mixed-Effects modeling:

Latency ~ Group * ACC + Age + Education + (1|SubjectID)

5. Kernel Density Estimation (KDE) of latency distributions
6. Leave-One-Subject-Out (LOSO) classification
- Latency-based
- Beta difference–based
- Combined model
- Permutation-based AUC testing

---

## 📦 Data Availability

The preprocessed dataset (compressed, 65MB) is included in:

 Stroop_AllSubjects_NIRS_HRF_TTest_05.zip

All subjects are anonymized and identified using research codes.  
No personally identifiable information is included.

---

## 🔁 Reproducing the Main Results

### Step 1 — Unzip dataset

Extract:


Stroop_AllSubjects_NIRS_HRF_TTest_05.zip

Place the extracted `.mat` file inside:

---

### Step 2 — Run latency analysis

In MATLAB:

```matlab
A2_LMEandKDE

This generates:

Trial-level latency table

LME results

KDE distributions

Step 3 — Run classification
A6_run_loso_full

This performs:

Pairwise HC vs MCI vs AD classification

LOSO logistic regression

Permutation-based AUC significance testing

🛠 Requirements

MATLAB (R2020b or later recommended)

Statistics and Machine Learning Toolbox

Signal Processing Toolbox

SPM (for spm_hrf) added to MATLAB path

👤 Maintainer

Sohyeon Yoo
GitHub: https://github.com/Sohyeon9367
