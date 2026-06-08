# Model Card — CardioRisk AI

> Generated following the [Google Model Card framework](https://modelcards.withgoogle.com/about) and TRIPOD-AI reporting standards.

---

## 1. Model Details

| Field | Value |
|---|---|
| **Model Name** | CardioRisk AI |
| **Version** | 2.0 |
| **Date Released** | June 2026 |
| **Model Type** | Calibrated Stacking Ensemble (binary classifier) |
| **Owner** | HayMedics Academy |
| **License** | MIT |
| **Reporting Standard** | TRIPOD-AI (Collins et al., 2024) |
| **Code Repository** | [github.com/HayMedics/cardiorisk-ai](https://github.com/HayMedics/cardiorisk-ai) |

### Architecture
- **Level 1 (Base learners):** Gradient Boosting, Logistic Regression, Random Forest, Extra Trees
- **Level 2 (Meta-learner):** Logistic Regression (5-fold CV)
- **Calibration:** Isotonic regression (3-fold CV)
- **Output:** Calibrated probability of CAD (≥50% stenosis in any major vessel)

---

## 2. Intended Use

### Primary intended uses
- **Research & education** — demonstrating an end-to-end clinical ML pipeline
- **Pedagogy** — teaching TRIPOD-AI reporting, calibration, cost-sensitive thresholds
- **Portfolio demonstration** — showcasing healthcare ML deployment skills

### Out-of-scope uses
- ❌ **NOT** for clinical decision-making
- ❌ **NOT** for diagnosis, treatment selection, or patient management
- ❌ **NOT** validated for use in any contemporary clinical setting
- ❌ **NOT** to be deployed in any EHR or clinical workflow

### Intended users
- Data science learners
- Healthcare AI researchers (for benchmarking)
- Clinical informaticians (for teaching demos)

---

## 3. Training Data

| Field | Details |
|---|---|
| **Source** | UCI Machine Learning Repository — Cleveland Heart Disease |
| **Collection Period** | 1981–1984 |
| **Institution** | Cleveland Clinic Foundation, USA |
| **Original Reference** | Detrano R et al. *Am J Cardiol.* 1989;64(5):304-310 |
| **Sample Size** | 302 patients (after deduplication) |
| **Prevalence** | ~46% positive class (CAD) after label correction |
| **Class Balance** | Mildly imbalanced (handled via SMOTE-NC within CV folds) |

### Data preprocessing
- Exact duplicates removed
- `ca=4` recoded as missing (procedural artifact per dataset documentation)
- Median imputation for continuous features; mode for categorical
- IQR-based winsorisation for outliers
- RobustScaler for continuous features (resistant to remaining outliers)
- OneHotEncoder for categorical features

### Train/validation/test split
- 60% train (n=181) / 20% validation (n=60) / 20% test (n=61)
- Stratified by outcome
- Indices saved to `splits.json` for reproducibility

---

## 4. Evaluation Data

Same dataset as training (single-source). Test set held out from all preprocessing fitting and hyperparameter tuning.

> ⚠️ **External validation has NOT been performed.** Model performance on patients outside the 1981–1984 Cleveland Clinic population is unknown.

---

## 5. Performance Metrics

### Test set performance (n=61)

| Metric | Value | 95% CI* |
|---|---|---|
| AUC-ROC | 0.8745 | 0.78–0.94 |
| Sensitivity (Recall) | 1.0000 | 0.89–1.00 |
| Specificity | 0.5152 | 0.34–0.69 |
| PPV (Precision) | 0.6364 | 0.50–0.76 |
| NPV | 1.0000 | 0.81–1.00 |
| F2-score | 0.8911 | — |
| Brier score | 0.139 | — |

*95% CIs estimated via 2000 bootstrap resamples

### Threshold selection
- **Method:** F2-score maximisation (favours sensitivity)
- **Final threshold:** 0.18
- **Rationale:** Clinical cost matrix (FN = 10 × FP penalty)

### Calibration
Isotonic calibration applied. Brier score = 0.139 (lower is better).

---

## 6. Ethical Considerations

### Fairness
- ⚠️ **No ethnicity or race data** in source — cannot assess racial fairness
- ⚠️ **Sample is geographically homogeneous** (single US Midwest hospital)
- ⚠️ **No subgroup analysis** performed for socioeconomic status, BMI, or comorbidities

### Privacy
- ✅ Source data is fully anonymised and public domain
- ✅ No patient identifiers used in model development or deployment
- ✅ No personal data collected by the deployed application

### Potential harms if misused
- **False reassurance:** Low-risk predictions could delay needed cardiac investigation
- **Over-referral:** False positives could increase healthcare costs and patient anxiety
- **Diagnostic anchoring:** Clinicians may over-rely on model output and skip clinical judgement
- **Equity gap:** Performance on under-represented demographics is unknown

### Mitigation
- Explicit "Research Only" disclaimers in app UI, README, model card, and PDF reports
- High sensitivity threshold prioritises catching disease over false alarms
- Per-prediction feature explanations encourage clinical scrutiny
- Open-source code allows community audit

---

## 7. Caveats & Recommendations

### Known caveats
1. Training data is **40+ years old** — clinical practice has evolved significantly
2. **No medication data** — beta-blockers affect heart rate features
3. **No modern biomarkers** — troponin, NT-proBNP, CT calcium are absent
4. **Single institution** — Cleveland Clinic only
5. **Small sample** (n=302) — limits subgroup analyses
6. **No temporal/longitudinal data** — single time-point per patient
7. **Label artifact in public dataset** — Kaggle version has inverted labels; this implementation includes a correction

### Recommendations for users
- Treat predictions as educational illustrations, not clinical guidance
- Always verify on clinical fundamentals (history, exam, ECG, biomarkers)
- For research use, consider modern alternatives: MIMIC-IV, UK Biobank, CARDIA
- If extending this work, perform external validation on contemporary cohorts

---

## 8. Quantitative Analyses

### Feature importance (perturbation-based, top 6)

| Feature | Mean |Δ probability| | Direction |
|---|---|---|
| ca (vessels stenosed) | 0.18 | ↑ Strong positive |
| oldpeak (ST depression) | 0.12 | ↑ Strong positive |
| thalach (max heart rate) | 0.09 | ↓ Strong negative |
| cp (chest pain type) | 0.08 | Variable |
| thal (perfusion result) | 0.07 | ↑ Positive |
| slope (ST slope) | 0.06 | Variable |

> These align with established cardiology literature on CAD risk markers.

### Clinical sanity checks (performed during development)

| Test Patient Profile | Expected | Model Output | Pass |
|---|---|---|---|
| 65yo male, asymptomatic, oldpeak=3.5, ca=3 | HIGH | 75.6% | ✅ |
| 35yo female, non-anginal, oldpeak=0, ca=0 | LOW | 5.6% | ✅ |

---

## 9. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | June 2026 | Initial release: stacking ensemble + Streamlit UI |
| 1.1 | June 2026 | Label correction (Kaggle inverted target artifact) |
| 2.0 | June 2026 | Added SHAP-style explanations + PDF reports + What-If simulator |

---

## 10. Contact

For questions about this model card or the underlying model:

- 🌐 **Live demo:** [cardiorisk-ai-haymedics.streamlit.app](https://cardiorisk-ai-haymedics.streamlit.app)
- 💻 **Repository:** [github.com/HayMedics/cardiorisk-ai](https://github.com/HayMedics/cardiorisk-ai)
- 💼 **LinkedIn:** www.linkedin.com/in/
awal-abdulrahman-md-a53483219


---

> ⚠️ **Final reminder:** This model is for research and educational purposes only.
> It is not a medical device. It must not be used to diagnose, treat, or manage any patient.

---

*Model card last updated: June 2026*
*Generated following TRIPOD-AI (Collins et al. 2024) and Google Model Card framework*
