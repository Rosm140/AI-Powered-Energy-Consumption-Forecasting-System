# ⚡ AI-Powered Energy Consumption Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

> Forecast hourly electricity consumption using machine learning — supporting smart cities, green buildings, and net-zero infrastructure.

---

## 📌 Problem Statement

Power grids fail when supply ≠ demand. This project builds an AI system that predicts future energy usage from historical smart-grid logs, enabling:

- **Grid operators** to balance load and prevent blackouts  
- **Buildings** to cut peak-hour penalties  
- **Smart cities** to plan renewable energy integration  
- **Industries** to reduce carbon emissions through optimized consumption

---

## 🏭 Industry Relevance

| Sector | Use Case |
|---|---|
| Smart Cities | Predict district-level demand for load balancing |
| Data Centers | Optimize cooling and server scheduling |
| Manufacturing | Shift high-energy tasks to off-peak hours |
| Renewable Energy | Align solar/wind output with predicted demand |
| Electricity Boards | Reduce wastage and generation costs |

**Companies actively hiring for this domain:** Google, Siemens, Schneider Electric, Tata Power, Microsoft, ABB, Honeywell, TCS, Infosys, Wipro.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn (Random Forest, Gradient Boosting, MLP) |
| Visualization | Matplotlib, Seaborn |
| Deployment | Flask REST API |
| Model Persistence | Joblib |

---

## 📁 Project Structure

```
AI-Energy-Forecasting/
│
├── data/
│   ├── energy.csv                  # Generated smart-grid dataset (17,520 rows)
│   └── generate_dataset.py         # Synthetic data generator
│
├── src/
│   ├── preprocess.py               # Data loading, cleaning & feature engineering
│   ├── train.py                    # Model training, evaluation & artifact saving
│   ├── visualize.py                # 6 publication-quality plots
│   └── app.py                      # Flask REST API with 4 endpoints
│
├── models/
│   ├── best_model.pkl              # Saved best model (Gradient Boosting)
│   └── scaler.pkl                  # Feature scaler
│
├── outputs/
│   ├── predictions.csv             # Test set predictions from all 3 models
│   ├── metrics.json                # Evaluation metrics (MAE, RMSE, R², MAPE)
│   └── plots/
│       ├── 01_energy_trend.png
│       ├── 02_usage_patterns.png
│       ├── 03_actual_vs_predicted.png
│       ├── 04_residuals.png
│       ├── 05_model_comparison.png
│       └── 06_correlation_heatmap.png
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb      # Full exploratory walkthrough
│
├── docs/
│   └── architecture.md             # System design & data flow
│
├── main.py                         # One-command pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip

### Setup (Windows / Mac / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AI-Energy-Forecasting.git
cd AI-Energy-Forecasting

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option A — Full Pipeline (Recommended)
```bash
python main.py --all
```
This will:
1. Generate a 2-year synthetic smart-grid dataset  
2. Train 3 models (Random Forest, Gradient Boosting, MLP)  
3. Save the best model  
4. Generate 6 visualizations  

### Option B — Individual Steps
```bash
python main.py --generate    # generate dataset
python main.py --train       # train models
python main.py --visualize   # generate plots
python main.py --api         # launch REST API
```

### Option C — REST API
```bash
python main.py --api
# API live at http://127.0.0.1:5000
```

**Single prediction:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"hour": 14, "day_of_week": 2, "month": 6}'
```

**Response:**
```json
{
  "predicted_energy_kWh": 47.312,
  "model": "GradientBoostingRegressor"
}
```

---

## 📊 Results

| Model | MAE (kWh) | RMSE (kWh) | R² | MAPE |
|---|---|---|---|---|
| Random Forest | 2.69 | 3.96 | 0.923 | 15.46% |
| **Gradient Boosting** | **2.55** | **3.80** | **0.929** | **14.39%** |
| MLP Neural Network | 2.56 | 3.81 | 0.928 | 14.17% |

> **Best model: Gradient Boosting** with R² = 0.929 — meaning the model explains 92.9% of the variance in energy consumption.

---

## 📸 Sample Outputs

### Energy Consumption Trend
![Trend](outputs/plots/01_energy_trend.png)

### Usage Patterns by Hour & Day
![Patterns](outputs/plots/02_usage_patterns.png)

### Actual vs Predicted
![Forecast](outputs/plots/03_actual_vs_predicted.png)

### Model Comparison
![Comparison](outputs/plots/05_model_comparison.png)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model-info` | Deployed model metadata & metrics |
| POST | `/predict` | Single prediction |
| POST | `/predict/bulk` | Batch predictions |

---

## 🧠 Features Engineered

| Feature | Description |
|---|---|
| `hour_sin / hour_cos` | Cyclical encoding of hour (0–23) |
| `day_sin / day_cos` | Cyclical encoding of day of week |
| `month_sin / month_cos` | Cyclical encoding of month (seasonal) |
| `is_weekend` | Binary flag for Saturday/Sunday |
| `lag_1h` | Energy usage 1 hour ago |
| `lag_24h` | Energy usage 24 hours ago (same hour yesterday) |
| `lag_168h` | Energy usage 168 hours ago (same hour last week) |
| `rolling_24h` | 24-hour rolling average |

---

## 📚 Learning Outcomes

After completing this project you will understand:
- Time-series feature engineering techniques
- Comparing multiple ML models on the same task
- Evaluating regression with MAE, RMSE, R², and MAPE
- Building and deploying a Flask prediction API
- Professional project structuring for GitHub

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙋 Author

**[Your Name]**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

> *"70% of India's power loss is due to poor forecasting. Let's fix that with AI."*
