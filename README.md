# Phishing Website Detection Using Machine Learning

> CAP6135 — Malware and Software Vulnerability Analysis
> University of Central Florida | Spring 2026
> **Authors:** Himasriya Engu (5753322) & Priyanka Basava (5754433)

---

## Overview
This study replicates and extends the work of Anakal et al. (IEEE ICIICS 2023) by implementing eight supervised machine learning models for phishing URL detection across two datasets.

---

## Reference Paper
Anakal et al., "Phishing Website Detection Using Machine Learning Methods,"
IEEE ICIICS 2023. DOI: 10.1109/ICIICS59993.2023.10420933

---

## Datasets
| Dataset | Samples | Features | Source |
|---------|---------|----------|--------|
| Kaggle Phishing | 11,054 | 31 | [Kaggle](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector) |
| Large Scale | 88,647 | 112 | [GitHub](https://github.com/GregaVrbancic/Phishing-Dataset) |

---

## Models Implemented
- Logistic Regression
- Random Forest ⭐ (Best Model)
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- AdaBoost
- Naive Bayes (Extension — not in reference paper)

---

## Results Summary

### Kaggle Dataset (11k)
| Model | Paper Accuracy | Our Accuracy | Our F1-Score |
|-------|---------------|-------------|-------------|
| Random Forest | 97.3% | 96.92% | 97.26% |
| Decision Tree | 96.6% | 96.02% | 96.43% |
| SVM | 95.3% | 94.98% | 95.58% |
| Gradient Boosting | 94.6% | 94.93% | 95.50% |
| KNN | 94.4% | 93.98% | 94.63% |
| AdaBoost | 93.5% | 93.89% | 94.57% |
| Logistic Regression | 92.8% | 93.35% | 94.12% |
| Naive Bayes* | N/A | 60.47% | 72.08% |

### Large Dataset (88k)
| Model | Accuracy |
|-------|---------|
| Random Forest | 97.02% |
| Decision Tree | 95.47% |
| Gradient Boosting | 95.34% |
| KNN* | 94.75% |
| Logistic Regression | 92.84% |
| AdaBoost | 92.41% |
| SVM* | 91.85% |

*Trained on 20k subset due to computational constraints

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Kaggle dataset (8 models)
```bash
python phishing_kaggle_model.py
```

### Run Large dataset (7 models)
```bash
python phishing_model.py
```

---

## Repository Structure
```
├── phishing_kaggle_model.py  # Kaggle dataset (11k) — 8 models
├── phishing_model.py         # Large dataset (88k) — 7 models
├── requirements.txt          # Required Python libraries
├── results/                  # All generated graphs and figures
│   ├── paper_vs_our_results.png
│   ├── kaggle_model_comparison.png
│   ├── kaggle_accuracy_ranking.png
│   ├── kaggle_confusion_matrix_rf.png
│   ├── large_model_comparison.png
│   ├── large_accuracy_ranking.png
│   ├── cross_dataset_comparison.png
│   ├── feature_importance.png
│   └── confusion_matrices.png
└── README.md
```

---

## Technologies Used
- **Language:** Python 3.9+
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, xgboost
- **Environment:** macOS
