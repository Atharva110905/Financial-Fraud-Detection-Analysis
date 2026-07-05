# Financial Fraud Detection Model & Dashboard

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. (Optional — already done) Regenerate data: `python generate_dataset.py`
3. (Optional — already done) Run ETL: `python etl_pipeline.py`
4. (Optional — already done) Train models: `python train_models.py`
5. Launch dashboard: `streamlit run app.py`

## Project structure
- `generate_dataset.py` — creates the synthetic 100K-row transaction dataset with 5 engineered fraud patterns
- `fraud_transactions.csv` — the generated dataset
- `etl_pipeline.py` — Extract/Transform/Load script that builds `fraud_detection.db`
- `fraud_detection.db` — SQLite database used by the dashboard
- `train_models.py` — trains and evaluates 7 models (Logistic Regression, Random Forest, XGBoost, LightGBM, Isolation Forest, One-Class SVM, Autoencoder)
- `artifacts/` — saved trained models, metrics, ROC/PR curve data, feature importance
- `app.py` — the Streamlit dashboard (5 tabs: Overview, EDA, Model Comparison, Transaction Investigator, ETL Pipeline)

## Fraud patterns simulated
1. Card Testing — rapid small-value transaction bursts
2. Account Takeover — high-value spend bursts after dormancy
3. Geo-Velocity Mismatch — transactions in distant cities within short time gaps
4. Odd-Hour High-Value — large transactions at unusual hours (12am-4am)
5. New Device + High-Risk Merchant — first-time device combined with high-risk category (crypto, gambling, money transfer)

## Model results summary (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 99.52% | 81.6% | 94.9% | 0.878 | 0.999 |
| LightGBM | 99.51% | 81.8% | 93.8% | 0.874 | 0.999 |
| Random Forest | 99.47% | 81.9% | 90.4% | 0.860 | 0.998 |
| Logistic Regression | 93.46% | 20.8% | 93.6% | 0.340 | 0.984 |
| Isolation Forest | 97.97% | 43.3% | 42.2% | 0.427 | 0.947 |
| One-Class SVM | 96.25% | 24.5% | 52.2% | 0.334 | 0.858 |
| Autoencoder | 94.11% | 14.3% | 45.6% | 0.218 | 0.783 |

Supervised tree-based models (XGBoost, LightGBM, Random Forest) perform best since they learn the exact fraud patterns from labeled data. Unsupervised models score lower but remain valuable for catching novel fraud types not seen in training.
