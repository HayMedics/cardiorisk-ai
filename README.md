# 🫀 CardioRisk AI

> A research-grade machine learning system for Coronary Artery Disease (CAD) risk classification, built to TRIPOD-AI reporting standards.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cardiorisk-ai-haymedics.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![TRIPOD-AI](https://img.shields.io/badge/Standard-TRIPOD--AI-1B3080?style=for-the-badge)](https://www.tripod-statement.org/)

**Built by:** HayMedics Academy · *Data | Research | Innovation*

🌐 **Live Demo:** [cardiorisk-ai-haymedics.streamlit.app](https://cardiorisk-ai-haymedics.streamlit.app)

---

## 🎯 Project Overview

**CardioRisk AI** estimates the probability of Coronary Artery Disease (≥50% stenosis in any major vessel) from 13 routine clinical features. The system uses a calibrated stacking ensemble with a cost-based decision threshold that penalises false negatives 10× more than false positives — because missing CAD is far costlier than an unnecessary follow-up referral.

This project answers a specific clinical question:

> *What if the model could see what the rushed clinician at the end of a 12-hour shift might miss?*

Not to replace the cardiologist, but to ensure every patient receives the same standard of risk stratification regardless of clinician fatigue or time pressure.

---

## ✨ Key Features

### v1 (Production)
- 🎯 **Cost-optimised binary classification** (CAD vs No CAD)
- 📊 **Calibrated probability output** for clinical interpretability
- 🚨 **Real-time clinical flags** (chronotropic incompetence, hypertensive crisis, high-risk ETT composite, etc.)
- 🎨 **Professional branded UI** with HayMedics Academy identity

### v2 (Latest)
- 🧠 **SHAP-style feature explanations** — see exactly which clinical factors drove the prediction
- 📄 **Downloadable PDF clinical reports** — A4 reports formatted for patient records
- 🎛️ **What-If Simulator** — explore how risk changes when variables are modified

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **AUC-ROC** | 0.8745 | Strong discrimination — comparable to published literature (range: 0.82–0.91) |
| **Sensitivity** | 100% | Catches all disease cases on test set (no false negatives) |
| **Specificity** | 51.5% | Trade-off accepted to maximise FN reduction |
| **PPV (Precision)** | 63.6% | 64% of positive predictions are true positives |
| **NPV** | 100% | A "low risk" prediction reliably excludes disease |
| **Decision Threshold** | 0.18 | Cost-optimised (FN cost = 10 × FP cost) |

> 📌 **Why this trade-off?** In clinical CAD screening, a missed diagnosis can be fatal. A false alarm leads to additional tests; a false reassurance can lead to a missed myocardial infarction. The threshold of 0.18 prioritises catching disease over avoiding follow-up referrals — aligned with ACC/AHA primary prevention philosophy.

---

## 📊 Dataset & Clinical Context

This project uses the **Cleveland Heart Disease dataset** (Detrano et al., 1989) — the most widely-cited cardiovascular ML benchmark, enabling direct comparison against decades of published literature.

### Why this dataset
- Gold-standard cardiology ML benchmark (3,000+ academic citations since 2000)
- Clean, structured, publicly available — ideal for end-to-end pipeline development
- Permits transparent benchmarking against published AUC values

### Acknowledged limitations (per TRIPOD-AI Item 19)
- 🗓️ **Temporal:** Patient data from 1981–1984 — demographics, diagnostic criteria, and treatment paradigms have evolved
- 🩸 **Biomarker gap:** No troponin, NT-proBNP, hsCRP, or CT coronary calcium scores
- 💊 **No medication data:** Beta-blockers, statins, and antihypertensives affect HR and lipid features but are not recorded
- 🏥 **Single centre:** Cleveland Clinic Foundation only — no geographic or institutional diversity
- 👥 **Sample size:** n=302 — limits subgroup analyses and external generalisability
- 🌍 **No ethnicity data:** Cannot assess fairness across racial/ethnic groups
- ⚖️ **Threshold artifact:** FBS uses dataset cutoff >120 mg/dL; current ADA standard is ≥126 mg/dL

### 🔍 Important Discovery During Development

The publicly available Kaggle version of this dataset has **inverted target labels** compared to the original UCI source. Patients labelled `target=1` are actually disease-free, while `target=0` represents diseased patients.

This was identified by examining mean feature distributions across classes:

| Feature | target=0 (claimed "no disease") | target=1 (claimed "disease") |
|---|---|---|
| oldpeak (ST depression) | 1.60 mm 🔴 HIGH | 0.57 mm ✅ LOW |
| thalach (max HR) | 139 bpm | 158 bpm |
| ca (vessels stenosed) | 1.16 🔴 MORE | 0.37 ✅ LESS |
| exang (exercise angina) | 55% 🔴 MORE | 13% ✅ LESS |

The "no disease" group had higher ST depression and more stenosed vessels than the "disease" group — clinically implausible. The model was retrained with corrected labels (`y_train_fixed = 1 - y_train`).

> 💡 **Practitioner note:** Always verify label orientation by examining feature distributions across classes before training. This artifact is easy to miss and has appeared in multiple published Kaggle notebooks.

### Roadmap to contemporary datasets

Future iterations will scale to modern cohorts:
- **MIMIC-IV** (~60,000 ICU stays, 2008–2019)
- **UK Biobank** (500,000+ participants with modern biomarkers)
- **CARDIA longitudinal study** (multi-ethnic, 30+ year follow-up)

---

## 🛠️ Methods (TRIPOD-AI Compliant)

### Data Pipeline
1. **Ingestion** — UCI/Kaggle Cleveland CSV (303 patients, 14 columns)
2. **Deduplication** — Drop exact row duplicates
3. **Missing data handling** — `ca=4` recoded as NaN (procedural artifact per data dictionary)
4. **Outlier treatment** — IQR-based winsorisation on continuous features (`trestbps`, `chol`, `thalach`)
5. **Train/Val/Test split** — 60/20/20 stratified by outcome, saved to `splits.json` for reproducibility

### Feature Engineering
**Clinical features (13):** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

**Engineered features (10):**
- `age_sex_interaction` — Sex-modified age risk
- `age_risk` — Binary threshold (Male ≥45, Female ≥55 per ACC/AHA 2019)
- `hr_reserve` — Chronotropic reserve capacity
- `pct_max_hr` — % of age-predicted maximum heart rate
- `chrono_incompetence` — Binary flag (HR <85% of predicted max)
- `high_risk_ett` — Composite (oldpeak ≥2.0 ∧ slope=downsloping ∧ exang=Yes)
- `bp_elevated` — Binary (trestbps ≥130 per AHA 2019)
- `chol_elevated` — Binary (chol ≥200 per ATPIII)
- `metabolic_score` — Sum of fbs + bp_elevated + chol_elevated
- `framingham_proxy` — Simplified Framingham risk approximation

### Preprocessing
- **Continuous:** Median imputation + RobustScaler (resistant to outliers)
- **Categorical:** Mode imputation + OneHotEncoder
- **Resampling:** SMOTE-NC applied **within** CV folds only (no test set leakage)
- ⚠️ **Critical:** Preprocessor fitted on training data only

### Model Architecture
Calibrated stacking ensemble:

```
┌─────────────────────────────────────────┐
│  Layer 1: Base Estimators (in parallel) │
├─────────────────────────────────────────┤
│  • Gradient Boosting (200 trees, d=4)   │
│  • Logistic Regression (L2, balanced)   │
│  • Random Forest (200 trees, d=6)       │
│  • Extra Trees (200 trees, d=6)         │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Layer 2: Meta-Learner                  │
├─────────────────────────────────────────┤
│  Logistic Regression (cv=5, balanced)   │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Calibration Layer                      │
├─────────────────────────────────────────┤
│  Isotonic Regression (cv=3)             │
└─────────────────┬───────────────────────┘
                  ↓
            Calibrated P(CAD)
```

### Validation Strategy
- **Internal:** 5-fold stratified cross-validation
- **Threshold selection:** F2-score optimisation on validation set (favours sensitivity)
- **Calibration:** Isotonic regression to ensure probabilities match observed frequencies
- **Cost matrix:** FN cost = 10 × FP cost (clinical CAD screening context)

---

## 🏗️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **ML** | scikit-learn (Stacking, CalibratedClassifierCV), imbalanced-learn (SMOTE-NC) |
| **Data** | pandas, numpy, scipy |
| **Visualisation** | matplotlib |
| **App Framework** | Streamlit 1.35 |
| **PDF Reports** | reportlab |
| **Deployment** | Streamlit Community Cloud + GitHub |
| **Standards** | TRIPOD-AI · ACC/AHA 2019 · ATPIII · ADA 2024 |

---

## 🚀 Quick Start

### Local installation

```bash
# Clone repository
git clone https://github.com/HayMedics/cardiorisk-ai.git
cd cardiorisk-ai

# Create environment
conda create -n cardiorisk python=3.11 -y
conda activate cardiorisk

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py
```

App will open at `http://localhost:8501`

### Try the live demo
👉 **[cardiorisk-ai-haymedics.streamlit.app](https://cardiorisk-ai-haymedics.streamlit.app)**

---

## 📁 Repository Structure

```
cardiorisk-ai/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version pin (3.11)
├── model.pkl                       # Trained stacking ensemble
├── preprocessor.pkl                # Fitted ColumnTransformer
├── feature_config.json             # Metrics & threshold metadata
├── heart_cad_research_v1.ipynb     # Full TRIPOD-AI research notebook
├── heart.csv                       # Cleveland dataset
├── HMA_ICON.jpg                    # HayMedics branding
├── HMA__Tagline_PNG.png            # HayMedics tagline logo
├── HayMedics_Academy06.jpg         # Full brand logo
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 📚 Clinical References

- **Dataset:** Detrano R, et al. International application of a new probability algorithm for the diagnosis of coronary artery disease. *Am J Cardiol.* 1989;64(5):304-310.
- **Reporting Standard:** Collins GS, et al. TRIPOD-AI and PROBAST-AI: prediction model reporting and risk of bias assessment guidelines for AI. *BMJ.* 2024.
- **BP Classification:** Whelton PK, et al. 2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults. *J Am Coll Cardiol.* 2018;71(19):e127-e248.
- **Cholesterol:** Grundy SM, et al. 2018 AHA/ACC Guideline on the Management of Blood Cholesterol. *J Am Coll Cardiol.* 2019;73(24):e285-e350.
- **Diabetes:** American Diabetes Association. Classification and Diagnosis of Diabetes: Standards of Medical Care 2024. *Diabetes Care.* 2024;47(Suppl 1):S20-S42.

---

## ⚠️ Medical Disclaimer

> **This software is for research and educational purposes only.**
> It is NOT a medical device. It has NOT been validated for clinical use.
> It must NOT be used to diagnose, treat, or manage any patient.
> Predictions are based on a small, single-centre, historical dataset (n=302, 1981–1984).
> Always consult a qualified healthcare professional for medical decisions.

---

## 🛣️ Roadmap

### v2.1 — Patient History & Comparison (planned)
- Session-based history of last 10 predictions
- Risk comparison chart vs population distribution

### v3.0 — Batch Processing (planned)
- CSV upload for bulk patient predictions
- Downloadable batch report
- Confidence intervals on each prediction

### v4.0 — Modern Cohort (research-grade)
- Migrate to MIMIC-IV (~60,000 ICU patients)
- Add troponin, NT-proBNP, CRP biomarkers
- External validation across hospital sites
- Time-series features (longitudinal vitals)

---

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome:

- 🐛 **Found a bug?** Open an issue
- 💡 **Have a suggestion?** Open an issue with the `enhancement` tag
- 🤝 **Want to collaborate?** Connect on LinkedIn

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The Cleveland Heart Disease dataset is in the public domain via the UCI Machine Learning Repository.

---

## 📬 Contact

**HayMedics Academy**
*Bridging Clinical Medicine and AI Research*

🌐 Live App: [cardiorisk-ai-haymedics.streamlit.app](https://cardiorisk-ai-haymedics.streamlit.app)
💼 LinkedIn: www.linkedin.com/in/
awal-abdulrahman

📧 Email: haymedicsacademy@gmail.com
---

<p align="center">
  <strong>🫀 Built with care for clinical AI excellence</strong><br>
  <em>Data · Research · Innovation</em>
</p>
