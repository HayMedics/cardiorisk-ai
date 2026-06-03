# CardioRisk AI 🫀

A research-grade machine learning system for Coronary Artery Disease (CAD) risk classification using the Cleveland Heart Disease dataset, built following TRIPOD-AI reporting standards.

**Built by:** [Your Name] | HayMedics Academy

🎯 Project Overview

CardioRisk AI estimates the probability of CAD (≥50% stenosis) from 13 clinical features. The system uses a stacking ensemble model with isotonic calibration and a cost-based decision threshold (FN penalised 10× FP).

📊 Model Performance

- **AUC-ROC:** 0.87
- **Sensitivity:** 100%
- **Specificity:** 52%
- **Threshold:** 0.18 (cost-optimised)

🔑 Key Discovery

During development, I identified that the publicly available Cleveland dataset has **inverted labels** compared to the original UCI source — target=1 corresponds to NO disease in the Kaggle version. Verified by examining mean feature values across classes (e.g., oldpeak: target=0 had higher ST depression than target=1, which is clinically backwards).

🛠 Tech Stack

- **Languages:** Python 3.11
- **ML:** scikit-learn (Stacking Ensemble), CalibratedClassifierCV
- **App:** Streamlit
- **Standards:** TRIPOD-AI

🏥 Clinical Features

age · sex · cp · trestbps · chol · fbs · restecg · thalach · exang · oldpeak · slope · ca · thal

🧠 Engineered Features

HR Reserve · % Max HR · Chronotropic Incompetence · Framingham Proxy · Metabolic Score · High-risk ETT composite · Age-sex interaction

🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

⚠️ Disclaimer

Research use only. Not validated for clinical decision-making. Always consult a qualified cardiologist.

📚 References

- Detrano R et al. Am J Cardiol. 1989;64(5):304-310
- Collins GS et al. TRIPOD-AI Statement, 2024
- ACC/AHA 2019 Primary Prevention Guidelines
