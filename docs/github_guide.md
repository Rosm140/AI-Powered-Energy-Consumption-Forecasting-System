# GitHub Publishing Guide

## 1. Create the Repository

1. Go to https://github.com/new
2. Repository name: `AI-Energy-Forecasting`
3. Description: `AI-powered electricity consumption forecasting using Gradient Boosting and MLP. Includes Flask REST API, 6 visualizations, and production-ready code.`
4. Set to **Public**
5. Do NOT initialize with README (you have one already)
6. Add tags: `machine-learning`, `energy`, `forecasting`, `flask`, `scikit-learn`, `smart-grid`, `time-series`, `python`

---

## 2. Push the Project

```bash
cd AI-Energy-Forecasting

git init
git add .
git commit -m "Initial commit: complete AI energy forecasting pipeline"

git remote add origin https://github.com/<your-username>/AI-Energy-Forecasting.git
git branch -M main
git push -u origin main
```

---

## 7-Day Commit Plan (for recruiter visibility)

| Day | What to commit | Commit message |
|---|---|---|
| Day 1 | `data/generate_dataset.py`, `data/energy.csv` | `feat: add synthetic smart-grid dataset generator` |
| Day 2 | `src/preprocess.py` | `feat: data preprocessing and feature engineering module` |
| Day 3 | `src/train.py`, `models/` | `feat: train Random Forest, Gradient Boosting, MLP models` |
| Day 4 | `outputs/metrics.json`, `outputs/predictions.csv` | `results: model evaluation MAE 2.55 kWh, R² 0.929` |
| Day 5 | `src/visualize.py`, `outputs/plots/` | `feat: add 6 production-quality visualization plots` |
| Day 6 | `src/app.py` | `feat: Flask REST API with /predict and /predict/bulk endpoints` |
| Day 7 | `README.md`, `notebooks/`, `docs/` | `docs: complete README with results, architecture, and usage guide` |

---

## Making the Repo Look Professional

- Pin the repository on your GitHub profile
- Add a preview image (`outputs/plots/03_actual_vs_predicted.png`) as the social preview  
  → Settings → Social Preview → Upload image
- Enable GitHub Pages for the docs/ folder if you add an HTML report
- Star and fork a few related repos to signal domain interest to recruiters
