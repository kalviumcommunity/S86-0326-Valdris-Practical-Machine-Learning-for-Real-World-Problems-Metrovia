# Metrovia — Practical Machine Learning for Real-World Transit Problems

**Analyst:** Valdris  
**Domain:** Public Transit Delay Prediction  
**Task Type:** Regression / Classification  
**Status:** Conceptual documentation — implementation pending

---

## Project Overview

Metrovia analyzes historical bus and metro transit data to predict delay durations and severity for upcoming trips. The goal is to alert commuters 30 minutes before a delay occurs, based on patterns learned from years of historical trip records.

**Example output:**
> Route 7, Stop 14, 8:45 AM Monday → 87% probability of delay > 5 minutes

The model does not guarantee outcomes. It says: based on everything learned from historical data, this is the most likely result.

---

## Repository Structure

```
metrovia/
├── data/
│   ├── raw/                        ← Original trip logs, weather records (immutable)
│   └── processed/                  ← Cleaned, feature-engineered records ready for modeling
├── notebooks/
│   └── exploratory_analysis.ipynb  ← Delay distribution plots, route-level EDA
├── src/
│   ├── data_preprocessing.py       ← Deduplication, missing value handling, format standardization
│   ├── feature_engineering.py      ← delay_minutes, hour_of_day, route_avg_delay_30d, etc.
│   ├── train.py                    ← Gradient boosting regressor, model serialization
│   └── evaluate.py                 ← RMSE, Precision, Recall, F1 on held-out test set
├── models/
│   └── delay_model.pkl             ← Serialized trained model artifact
├── requirements.txt                ← Pinned library versions
├── README.md                       ← This file
└── main.py                         ← Entry point that runs the full pipeline
```

`data/raw/` is treated as immutable — it is never overwritten. Processed data can always be regenerated from raw data using the preprocessing scripts. `src/` isolates each pipeline stage so each module is independently testable and replaceable. Notebooks are kept separate from source code because they cannot be imported, are hard to test, and are easy to run out of order.

---

## Problem Definition

| Property | Value |
|---|---|
| Target variable | `delay_minutes` (difference between scheduled and actual arrival) |
| Task type | Regression (predict exact delay) or Classification (on-time / minor / major) |
| Primary metric | RMSE for regression; Precision, Recall, F1 for classification |
| Business objective | Predict delays 30 minutes in advance to alert commuters |
| Training data | 18 months of historical trip records |
| Test data | Most recent 6 months (time-based split — not random) |

The time-based split is intentional. A random split on temporal data would allow the model to train on future trips and evaluate on past ones, which is a form of data leakage.

---

## ML Workflow

```
Raw Transit Data
      ↓
Data Cleaning        (remove noise, fix formats, handle missing values)
      ↓
Feature Engineering  (encode time, route history, weather signals)
      ↓
Train / Test Split   (18 months train → 6 months held out)
      ↓
Model Training       (gradient boosting regressor)
      ↓
Evaluation           (RMSE / F1 on held-out test set)
      ↓
Deployment           (live trip data → same feature pipeline → model prediction)
      ↓
Monitoring           (drift detection, retraining triggers)
```

Each stage feeds the next. A failure at any stage propagates forward — bad data produces bad features, bad features produce a bad model, a bad model produces harmful predictions.

---

## Data Cleaning

Raw transit data is messy by default: missing arrival times, duplicate trip records, inconsistent route naming, and outlier values from sensor errors.

Preprocessing steps:
- Remove duplicate trip records
- Impute or drop missing delay values
- Standardize route names and date/time formats
- Filter sensor noise and extreme outliers (e.g., recorded delays of 400 minutes)

Every trip record must have a valid scheduled time, actual time, route ID, and stop ID before moving to feature engineering.

---

## Feature Engineering

Feature engineering is where domain knowledge becomes model input. Raw columns are transformed into numerical signals the model can learn from.

| Raw Data | Engineered Feature | Why It Matters |
|---|---|---|
| Scheduled time & actual time | `delay_minutes` | The target variable |
| Timestamp | `hour_of_day`, `day_of_week`, `is_rush_hour` | Captures peak congestion patterns |
| Date | `is_weekend`, `is_holiday` | Different delay behavior on non-workdays |
| Route ID | `route_avg_delay_30d` | Encodes historical route-level performance |
| Stop sequence | `stop_position_ratio` | End-of-line stops accumulate more delay |
| Weather data | `is_raining`, `temperature_bin` | External factors that drive delays |
| Previous trip | `delay_last_trip` | Cascading delay signal |
| Rolling window | `route_avg_delay_7d` | Recent route performance without single-day noise |

Feature engineering consumes 60–80% of total project effort. A simple model with well-engineered features will outperform a complex model with poor features every time.

**Critical:** The exact same transformations applied during training must be applied at prediction time. Any mismatch causes silent, hard-to-debug errors in production.

---

## Model Training

**Algorithm:** Gradient boosting regressor  
**Training set:** 18 months of historical trip data  
**Test set:** Most recent 6 months (held out, never used during training)

The model learns a function that maps engineered features to predicted delay. During training it examines thousands of labeled historical trips, measures prediction error using a loss function, and adjusts its internal parameters to minimize that error. The output is a serialized model artifact saved to `models/delay_model.pkl`.

---

## Evaluation

The model is evaluated on the held-out 6-month test set — trips it has never seen during training.

**Regression metrics:**
- **RMSE** — root mean squared error; penalizes large errors more than small ones, appropriate when a 20-minute delay is much worse than a 2-minute delay

**Classification metrics:**
- **Precision** — of all trips flagged as delayed, how many actually were?
- **Recall** — of all trips that were actually delayed, how many did the model catch?
- **F1 Score** — harmonic mean of precision and recall; useful when both false positives (unnecessary alerts) and false negatives (missed delays) carry real cost

Accuracy alone is not reported. If 90% of trips run on time, a model that always predicts "on-time" achieves 90% accuracy and is completely useless. Precision and recall tell the real story.

---

## Failure Scenario: Data Leakage

**What goes wrong:** During feature engineering, `actual_arrival_time` is accidentally included as a training feature. This column is derived from the same source as the target variable (`delay_minutes`). The model learns to "predict" delays using information it would never have at real prediction time.

**What it looks like:** Near-perfect accuracy on the test set.

**What happens in production:** `actual_arrival_time` is unavailable for future trips (the trip hasn't happened yet), so it is dropped. Model performance collapses immediately after deployment.

**How to diagnose:** Compare features available at training time against features available at prediction time. Any feature derived from post-event data is a leakage risk. Always simulate the prediction-time data environment during evaluation.

---

## Monitoring

After deployment, the world keeps changing. Bus schedules change, new routes open, commuter patterns shift seasonally.

- **Data drift** — incoming trip data starts looking different from training data (e.g., a new fleet reduces average delays)
- **Concept drift** — the relationship between features and delays changes (e.g., a new express lane changes how `stop_position_ratio` relates to delay)

When drift is detected or performance drops below threshold, the model is retrained on recent data. A model that was accurate at launch can silently degrade over months without monitoring.

---

## Repository Analysis

### Strength: Explicit Data Leakage Awareness

The most technically mature aspect of this project is the dedicated failure scenario on data leakage. Rather than mentioning leakage as a vague warning, the README traces a specific, realistic failure from feature engineering through evaluation to production collapse — and provides a concrete diagnostic method. This reflects genuine ML engineering awareness that separates thoughtful practitioners from beginners focused only on accuracy numbers.

### Weakness: No Reproducibility Infrastructure

The most significant gap is the absence of reproducibility controls:

- No `requirements.txt` — no pinned library versions. The environment cannot be reconstructed by someone else.
- No `random_state` — gradient boosting uses internal randomness. Without a fixed seed, every training run produces slightly different results and specific metrics cannot be reproduced.
- No run instructions — the README explains the workflow conceptually but provides no commands. A new contributor cannot reproduce results without contacting the original author.
- No reported metrics — there are no actual RMSE or F1 values anywhere. Without numbers, there is nothing to evaluate or compare against.

**Fix:** Add `requirements.txt` with pinned versions, set `random_state=42` on the model and train/test split, add a "How to Run" section with exact commands, and report actual evaluation results with a naive baseline for comparison.

---

## How to Run

**Requirements:** Python 3.10+

**Setup (first time):**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**Run the pipeline:**
```bash
python main.py
```

**Deactivate when done:**
```bash
deactivate
```

---

## Key Principles

- Models do not understand business meaning — everything must be encoded as numbers
- Feature engineering determines model success more than algorithm choice
- Always evaluate on data the model has never seen
- The training feature pipeline and the serving feature pipeline must be identical
- Monitoring is not optional — the world changes, and models must adapt
- When a model fails, look at the data and features first, not the algorithm
