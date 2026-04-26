# Medical Informatics: Stochastic Representation of Healthcare Costs
**Srijan S. Kotian | Computer Science & Engineering**

##  Research Abstract
This project explores the intersection of ensemble learning and healthcare economics. By utilizing a **Random Forest Regressor**, the system transitions from point-estimation to **uncertainty quantification** in medical cost forecasting. The primary objective is to demonstrate a robust, interpretable framework for clinical decision support.

##  Methodology & Statistical Rigor
- **Algorithm:** Random Forest Ensemble (100 Decision Trees).
- **Validation:** 5-Fold Cross-Validation to ensure model generalization across heterogeneous patient cohorts.
- **Ethics & Bias Mitigation:** Implemented **One-Hot Encoding** for categorical variables (Region/Sex) to prevent ordinal bias.
- **Uncertainty Quantification:** Calculated the standard deviation across ensemble estimators to provide a confidence interval (CI) for every prediction.

## 📊 Performance Benchmarks
- **Mean Absolute Error (MAE):** $2,549.34 (Average deviation in premium estimation).
- **R² Score:** 0.8652 (Strong correlation between biomarkers and costs).
- **Cross-Validation Accuracy:** 83.60% (± 6.49% variance), ensuring model stability across cohorts.

## ⚠️ Limitations & Future Work
1. **Domain Gap:** The current model is trained on US-centric data; future iterations require fine-tuning on European (GDPR-compliant) datasets.
2. **Feature Scarcity:** Inclusion of longitudinal biomarkers and diagnostic history would reduce prediction variance.
3. **Model Selection:** Exploring Gradient Boosting (XGBoost) vs. Neural Networks for latent feature extraction.