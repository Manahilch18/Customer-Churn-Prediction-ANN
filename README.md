<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-1.25-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🏦 Customer Churn Prediction using Artificial Neural Networks</h1>

<p align="center">
  <em>A production-grade, cost-sensitive deep learning system that predicts bank customer churn<br>and ranks at-risk customers for targeted retention interventions.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-key-results">Key Results</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-methodology">Methodology</a>
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Key Results](#-key-results)
- [Dataset Description](#-dataset-description)
- [Methodology](#-methodology)
  - [Exploratory Data Analysis (EDA)](#1--exploratory-data-analysis-eda)
  - [Feature Engineering](#2--feature-engineering)
  - [Data Preprocessing](#3--data-preprocessing)
  - [Model Architecture & Training](#4--model-architecture--training)
- [Model Comparison](#-model-comparison)
- [Best Model & Justification](#-best-model--justification)
- [Performance Metrics](#-performance-metrics)
- [Business Impact & Cost Analysis](#-business-impact--cost-analysis)
- [Key Insights](#-key-insights)
- [Live Demo](#-live-demo)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Project Overview

Customer churn — when a customer stops doing business with a company — is one of the most expensive problems in the banking industry. Acquiring a new customer costs **5–25× more** than retaining an existing one. This project builds an end-to-end **Artificial Neural Network (ANN)** pipeline that:

1. **Predicts** which customers are most likely to churn  
2. **Optimizes** the decision threshold using a real-world cost function (not just accuracy)  
3. **Ranks** customers by risk probability for capacity-constrained intervention teams  
4. **Deploys** an interactive Streamlit dashboard for single & batch predictions  

> [!IMPORTANT]
> This is **not** a standard classification project. The model is deliberately tuned for **high recall** (88.7%) at the cost of lower precision, because missing a churner ($500 cost) is 10× more expensive than a false alarm ($50 cost).

---

## 🔍 Problem Statement

A multinational bank observed rising customer attrition across its European branches (France, Germany, Spain). The retention team can proactively intervene with **at most 500 customers per quarter** due to staffing constraints. The business needs a system that:

- **Accurately identifies churners** before they leave (minimize False Negatives)  
- **Ranks all customers by risk** so the limited intervention budget targets the right people  
- **Minimizes total business cost** using an asymmetric cost function  

| Scenario | Cost per Case | Description |
|:---------|:-------------|:------------|
| ❌ False Negative (missed churner) | **$500** | Lost revenue, re-acquisition cost |
| ⚠️ False Positive (unnecessary outreach) | **$50** | Retention offer to a non-churner |
| ✅ True Positive / True Negative | **$0** | Correct prediction |

---

## 🏆 Key Results

<table>
<tr>
<td width="50%">

### 📊 Model Performance (Test Set)

| Metric | Value |
|:-------|------:|
| **ROC-AUC** | **0.8571** |
| Recall | 88.70% |
| F1-Score | 50.67% |
| Precision | 35.46% |
| Accuracy | 64.85% |
| Optimal Threshold | 0.260 |

</td>
<td width="50%">

### 💰 Business Impact

| KPI | Value |
|:----|------:|
| **Minimum Expected Cost** | **$55,850** |
| Cost-Optimal Threshold | 0.260 |
| Recall@500 (Capacity) | 68.6% |
| Precision@500 | 55.8% |
| Churners Captured (Top 500) | 279 / 407 |

</td>
</tr>
</table>

> [!NOTE]
> **Why is accuracy only 64.85%?** The threshold (0.260) is intentionally set far below the default 0.50 to **maximize recall**. This flags more customers as at-risk, lowering precision and accuracy — but the total dollar cost is minimized. The ROC-AUC of 0.857 confirms the model's strong discriminative ability regardless of threshold.

---

## 📊 Dataset Description

The dataset is the **[Bank Customer Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)** dataset from Kaggle, containing records of 10,000 European bank customers.

| Property | Detail |
|:---------|:-------|
| **Records** | 10,000 customers |
| **Features** | 14 columns (10 used after preprocessing) |
| **Target Variable** | `Exited` (1 = Churned, 0 = Retained) |
| **Class Distribution** | 20.37% Churned · 79.63% Retained |
| **Imbalance Ratio** | ~1 : 3.9 |
| **Geography** | France, Germany, Spain |

### Feature Dictionary

| Feature | Type | Description |
|:--------|:-----|:------------|
| `CreditScore` | Numeric | Customer's credit score (350–850) |
| `Geography` | Categorical | Country of residence (France / Germany / Spain) |
| `Gender` | Binary | Male (1) / Female (0) |
| `Age` | Numeric | Customer's age in years |
| `Tenure` | Numeric | Years as a bank customer (0–10) |
| `Balance` | Numeric | Account balance in USD |
| `NumOfProducts` | Numeric | Number of bank products used (1–4) |
| `HasCrCard` | Binary | Whether the customer has a credit card |
| `IsActiveMember` | Binary | Whether the customer is an active member |
| `EstimatedSalary` | Numeric | Estimated annual salary in USD |
| `Exited` | Binary | **Target** — 1 if the customer churned |

> **Dropped columns**: `RowNumber`, `CustomerId`, `Surname` — identifiers with no predictive value.

---

## 🔬 Methodology

The project follows a structured, notebook-driven pipeline with 7 phases, each documented in a dedicated Jupyter notebook.

### 1. 📈 Exploratory Data Analysis (EDA)

**Notebook**: `notebooks/03_Exploratory_Data_Analysis.ipynb`

Key analyses performed:
- **Class imbalance visualization** — 20.37% positive class (churned) confirmed; guided the decision to use class-weighted training  
- **Univariate distributions** — Age, Balance, and NumOfProducts showed strong separation between churners and non-churners  
- **Correlation analysis** — No severe multicollinearity detected among features; safe for neural network training  
- **Geographic breakdown** — Germany exhibited a significantly higher churn rate (~32%) compared to France (~16%) and Spain (~17%)  
- **Demographic patterns** — Customers aged 40–60 showed disproportionately high churn rates  

**Key EDA Findings**:
- 🔴 **Age** is the strongest individual predictor of churn  
- 🔴 **Germany-based** customers churn at nearly double the rate of France/Spain  
- 🔴 Customers with **only 1 product** or **3+ products** are at elevated risk  
- 🟢 **Tenure** has minimal correlation with churn — long-tenured customers churn at similar rates  
- 🟢 **CreditScore** and **EstimatedSalary** show weak individual predictive power  

### 2. ⚙️ Feature Engineering

**Notebook**: `notebooks/04_Data_Preprocessing.ipynb`

| Transformation | Details |
|:---------------|:--------|
| One-Hot Encoding | `Geography` → `Geography_Germany`, `Geography_Spain` (drop-first encoding, France as baseline) |
| Label Encoding | `Gender` → 0 (Female) / 1 (Male) |
| Feature Selection | Dropped `RowNumber`, `CustomerId`, `Surname` (non-informative identifiers) |
| Feature Persistence | Final feature column order saved to `models/feature_columns.pkl` for inference consistency |

> [!TIP]
> Drop-first encoding was used for `Geography` to avoid the **dummy variable trap** (perfect multicollinearity in linear layers). With `Geography_Germany = 0` and `Geography_Spain = 0`, the model implicitly infers "France."

### 3. 🧹 Data Preprocessing

| Step | Implementation |
|:-----|:---------------|
| **Train/Test Split** | 80/20 stratified split (`stratify=y`, `random_state=42`) |
| **Scaling** | `StandardScaler` fit **only** on training data, applied to both train and test |
| **Scaler Persistence** | Saved to `models/scaler.pkl` for consistent inference transforms |
| **Reproducibility** | Global seed (`SEED = 42`) set for Python, NumPy, and TensorFlow |

```
Training set:   8,000 samples (1,630 churned · 6,370 retained)
Test set:        2,000 samples (  407 churned · 1,593 retained)
```

### 4. 🧠 Model Architecture & Training

**Script**: `src/train.py`

The production ANN architecture:

```
Input Layer (11 features)
    │
    ▼
Dense(64, ReLU) + He Normal Init + L2(0.001)
    │── BatchNormalization
    │── Dropout(0.30)
    ▼
Dense(32, ReLU) + He Normal Init + L2(0.001)
    │── BatchNormalization
    │── Dropout(0.20)
    ▼
Dense(16, ReLU) + He Normal Init
    │
    ▼
Dense(1, Sigmoid) → Churn Probability [0, 1]
```

| Hyperparameter | Value | Rationale |
|:---------------|:------|:----------|
| Optimizer | Adam | Adaptive learning rate, fast convergence |
| Learning Rate | 0.001 (initial) | Reduced on plateau (factor=0.5, patience=5) |
| Loss Function | Binary Cross-Entropy | Standard for binary classification |
| Batch Size | 32 | Balance between gradient noise and stability |
| Max Epochs | 150 | With early stopping (patience=15) |
| Class Weights | Balanced (computed via `sklearn`) | Compensate for 1:3.9 class imbalance |
| Regularization | L2 (λ=0.001) + Dropout (30%/20%) | Prevent overfitting |
| Weight Init | He Normal | Optimal for ReLU activations |

**Training Callbacks**:
- `EarlyStopping` — Monitors `val_loss`, stops after 15 epochs without improvement, restores best weights  
- `ReduceLROnPlateau` — Halves learning rate after 5 epochs of stagnation (min LR: 1e-6)  

---

## 📈 Model Comparison

Three training strategies were evaluated during development:

| Model Variant | Approach | ROC-AUC | Recall | Precision | F1-Score | Notes |
|:-------------|:---------|:-------:|:------:|:---------:|:--------:|:------|
| **Baseline ANN** | No class weights, default threshold (0.50) | ~0.84 | ~47% | ~76% | ~58% | High precision but misses half of all churners |
| **Class-Weighted ANN** | Balanced class weights, default threshold (0.50) | ~0.85 | ~72% | ~50% | ~59% | Better recall, improved balance |
| **Production ANN** ✅ | Class weights + cost-optimal threshold (0.26) | **0.857** | **88.7%** | **35.5%** | **50.7%** | Maximizes business value |

**5-Fold Stratified Cross-Validation** was also performed to verify model robustness and guard against overfitting. Results were saved to `reports/cross_validation_results.csv`.

---

## 🏅 Best Model & Justification

**✅ Selected Model: Production ANN (Class-Weighted + Cost-Optimal Threshold)**

The production model was chosen based on **business-value optimization**, not just statistical metrics:

1. **Cost Minimization** — At threshold 0.260, total expected cost is **$55,850** — the minimum across all evaluated thresholds  
2. **High Recall (88.7%)** — Captures nearly 9 out of 10 actual churners, directly reducing the $500/case false-negative cost  
3. **Capacity Efficiency** — When the bank can only contact 500 customers, the model captures **279 of 407 churners (68.6%)** in the top-ranked segment  
4. **Robust Discrimination** — ROC-AUC of **0.857** demonstrates strong class separation independent of threshold choice  

> [!IMPORTANT]
> A model with higher accuracy (e.g., 86%) at the default 0.50 threshold would **miss over half of all churners**, costing the bank an additional ~$100,000+ in lost customers per evaluation cycle.

---

## 📊 Performance Metrics

### Classification Report (Threshold = 0.260)

| Class | Precision | Recall | F1-Score | Support |
|:------|:---------:|:------:|:--------:|--------:|
| Retained (0) | **95.3%** | 58.8% | 72.7% | 1,593 |
| Churned (1) | **35.5%** | 88.7% | 50.7% | 407 |
| **Weighted Avg** | **83.1%** | **64.9%** | **68.2%** | **2,000** |

### Confusion Matrix

```
                  Predicted
                 Retained  Churned
Actual Retained    936       657
Actual Churned      46       361
```

### Capacity-Based Risk Ranking (Recall@K)

| Intervention Capacity (K) | Churners Captured | Recall@K | Precision@K |
|:------------------------:|:-----------------:|:--------:|:-----------:|
| 200 | 167 / 407 | 41.0% | 83.5% |
| 300 | 220 / 407 | 54.1% | 73.3% |
| **500** | **279 / 407** | **68.6%** | **55.8%** |
| 700 | 319 / 407 | 78.4% | 45.6% |
| 1,000 | 356 / 407 | 87.5% | 35.6% |

---

## 💰 Business Impact & Cost Analysis

### Cost-Sensitive Threshold Optimization

Instead of using the default 0.50 threshold, the model evaluates every threshold from 0.05 to 0.95 and selects the one that **minimizes total expected business cost**:

$$\text{Total Cost} = (FN \times \$500) + (FP \times \$50)$$

| Parameter | Value |
|:----------|:------|
| Cost of False Negative (missed churner) | $500 |
| Cost of False Positive (unnecessary outreach) | $50 |
| **Optimal Threshold** | **0.260** |
| **Minimum Expected Cost** | **$55,850** |

### Why This Matters

At the default threshold (0.50):
- Higher accuracy (~86%), but **Recall drops to ~47%** → the bank misses ~215 churners  
- Missed churner cost: 215 × $500 = **$107,500** in lost revenue  

At the optimal threshold (0.260):
- Lower accuracy (64.9%), but **Recall rises to 88.7%** → only ~46 churners missed  
- Additional false positive cost is only 657 × $50 = $32,850  
- **Net savings: ~$50,000+** per evaluation cycle  

---

## 💡 Key Insights

1. **Threshold tuning is more impactful than model architecture** — Moving from 0.50 to 0.26 saved $50,000+ per cycle without any change to the neural network itself  
2. **Class imbalance must be addressed at multiple levels** — Both class weights (during training) and threshold adjustment (during inference) were necessary  
3. **Age is the dominant predictor** — Customers aged 40–60 churn at dramatically higher rates; the bank should prioritize retention campaigns for this demographic  
4. **Geography matters** — German customers churn at ~2× the rate of French/Spanish customers, suggesting market-specific retention strategies  
5. **Product diversification is a double-edged sword** — Customers with 3–4 products have the highest churn rates, possibly due to service complexity  
6. **Capacity-based ranking > binary classification** — Sorting by probability (not just yes/no) allows the bank to allocate limited resources optimally  

---

## 🖥️ Live Demo

The project includes an interactive **Streamlit Dashboard** with four tabs:

<img width="1897" height="893" alt="Screenshot 2026-07-28 183431" src="https://github.com/user-attachments/assets/8bc3f55f-f290-460e-ae1a-b6e7b8b464b6" />


| Tab | Feature | Description |
|:----|:--------|:------------|
| 🎯 **Single Prediction** | Real-time inference | Enter customer details via sliders/dropdowns and get instant churn probability |
| 📋 **Batch Risk Ranking** | CSV upload & rank | Upload a CSV of customers, set intervention capacity (K), download ranked results |
| 📊 **Model Insights** | Performance reports | View metrics table, confusion matrix, ROC curve, and Precision-Recall curve |
| ℹ️ **About** | Documentation | Business assumptions, cost parameters, and project features |

### Dashboard Preview

```bash
# Launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` with a clean, card-based UI featuring:
- **KPI cards** showing total customers, flagged risk count, and estimated churners  
- **Interactive sliders** for credit score, age, tenure, and financial inputs  
- **One-click CSV download** for the ranked at-risk customer list  

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|:---------|:-----------|:--------|
| **Language** | Python 3.10+ | Core development language |
| **Deep Learning** | TensorFlow / Keras 2.13 | ANN model building, training, and inference |
| **ML Toolkit** | scikit-learn 1.3 | Preprocessing, metrics, cross-validation, class weights |
| **Data Processing** | Pandas 2.0 · NumPy 1.24 | Data manipulation and numerical operations |
| **Visualization** | Matplotlib 3.7 · Seaborn 0.12 | Static plots (ROC, PR curves, confusion matrix) |
| **Dashboard** | Streamlit 1.25 | Interactive web application for predictions |
| **Serialization** | Joblib 1.3 | Model artifact persistence (scaler, encoder, columns) |
| **Notebooks** | Jupyter | Exploratory analysis and documentation |

---

## 📁 Project Structure

```
Customer-Churn-Prediction-ANN/
│
├── 📄 app.py                          # Streamlit dashboard (4-tab UI)
├── 📄 requirements.txt                # Pinned Python dependencies
├── 📄 README.md                       # This file
├── 📄 .gitignore                      # Git ignore rules
│
├── 📂 src/                            # Core Python package
│   ├── config.py                      # Centralized paths, thresholds & cost constants
│   ├── train.py                       # Training pipeline (baseline, weighted, CV, production)
│   ├── evaluate.py                    # Evaluation, cost optimization & report generation
│   ├── predict.py                     # Single & batch prediction logic
│   └── utils.py                       # Artifact loading & risk-ranking utilities
│
├── 📂 notebooks/                      # Jupyter analysis notebooks (ordered pipeline)
│   ├── 01_Business_Understanding.ipynb
│   ├── 02_Data_Understanding.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Data_Preprocessing.ipynb
│   ├── 05_ANN_Model.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   └── 07_Hyperparameter_Tuning.ipynb
│
├── 📂 data/
│   ├── raw/                           # Original dataset (Churn_Modelling.csv)
│   └── proccessed/                    # Train/test splits (scaled)
│
├── 📂 models/                         # Serialized model artifacts
│   ├── ann_churn_production.h5        # Production ANN weights
│   ├── ann_churn_baseline.h5          # Baseline ANN weights
│   ├── scaler.pkl                     # StandardScaler (fit on raw X_train)
│   ├── label_encoder.pkl              # Label encoder for Gender
│   ├── feature_columns.pkl            # Ordered feature column names
│   ├── best_threshold.pkl             # Cost-optimal threshold (0.260)
│   └── business_capacity.pkl          # Default intervention capacity
│
└── 📂 reports/                        # Generated evaluation outputs
    ├── metrics.csv                    # Performance metrics summary
    ├── confusion_matrix.csv           # Confusion matrix values
    ├── capacity_analysis.csv          # Recall@K for various capacities
    └── figures/                       # Static plots
        ├── confusion_matrix.png
        ├── roc_curve.png
        └── precision_recall_curve.png
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher  
- pip (Python package manager)  
- Git  

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Manahilch18/Customer-Churn-Prediction-ANN.git
cd Customer-Churn-Prediction-ANN

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} OK')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} OK')"
```

---

## 💻 Usage

### 🖥️ Launch the Dashboard

```bash
streamlit run app.py
```


### 🔄 Retrain the Model

```bash
# Train from raw data → saves model, scaler, and processed splits
python -m src.train

# Evaluate on test set → generates metrics, plots, and capacity analysis
python -m src.evaluate
```

### 🐍 Use Programmatically

```python
from src.predict import predict_single, predict_batch
import pandas as pd

# Single customer prediction
result = predict_single({
    "CreditScore": 619,
    "Gender": 0,              # 0 = Female, 1 = Male
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
    "Geography_Germany": 0,
    "Geography_Spain": 0,      # France (both 0)
})
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Predicted Churn: {'Yes' if result['predicted_churn'] else 'No'}")

# Batch prediction from CSV
df = pd.read_csv("your_customers.csv")
ranked = predict_batch(df).sort_values("churn_probability", ascending=False)
print(ranked.head(10))
```

---

## 📊 Results

### Model Validation Summary

| Evaluation Aspect | Result |
|:-------------------|:-------|
| **Discriminative Power** | ROC-AUC = 0.857 (strong) |
| **Churner Detection** | Recall = 88.7% at optimal threshold |
| **Cost Efficiency** | $55,850 expected cost (minimized) |
| **Resource Targeting** | Top 500 captures 68.6% of all churners |
| **Cross-Validation** | 5-fold stratified CV confirms model stability |
| **Generalization** | No signs of overfitting (EarlyStopping + L2 + Dropout) |

### ROC & PR Curves

The model produces well-separated ROC and Precision-Recall curves, confirming that the high performance is not due to chance or data leakage. All evaluation charts are stored in `reports/figures/`.

---

## 🔮 Future Improvements

| Priority | Improvement | Expected Impact |
|:--------:|:------------|:----------------|
| 🔴 High | **SHAP / LIME explainability** — Add per-prediction feature importance to the dashboard | Builds trust with business stakeholders |
| 🔴 High | **Feature engineering** — Derive interaction features (e.g., `Balance × NumOfProducts`, `Age × IsActiveMember`) | Potential 2–5% AUC lift |
| 🟡 Medium | **Ensemble methods** — Compare with XGBoost, LightGBM, or a stacking ensemble | Tree-based models often outperform ANNs on tabular data |
| 🟡 Medium | **Temporal modeling** — Incorporate customer activity trends over time (RNN / LSTM) | Captures churn trajectory, not just a snapshot |
| 🟡 Medium | **MLflow integration** — Track experiments, hyperparameters, and model versions systematically | Production-grade experiment management |
| 🟢 Low | **Docker deployment** — Containerize the Streamlit app for cloud deployment | One-command deployment to AWS/GCP/Azure |
| 🟢 Low | **API endpoint** — Wrap prediction logic in a FastAPI REST service | Enable integration with existing bank CRM systems |
| 🟢 Low | **CI/CD pipeline** — Automated testing, linting, and model retraining on new data | Engineering best practices |

---

## 👤 Author

**Manahil Ishfaq**

<p>
  <a href="https://github.com/Manahilch18"><img src="https://img.shields.io/badge/GitHub-Manahilch18-181717?style=flat-square&logo=github" alt="GitHub"></a>
</p>

If this project helped you or you found it interesting, please consider giving it a ⭐ — it helps more people discover it!

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — free for personal, academic, and commercial use.
```

---

<p align="center">
  <strong>Built with streamlit and TensorFlow</strong><br>
  <em>Turning data into actionable retention strategies</em>
</p>
