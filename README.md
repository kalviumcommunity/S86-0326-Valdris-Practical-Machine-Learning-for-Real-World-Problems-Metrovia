# Metrovia — Practical Machine Learning for Real-World Transit Problems

Metrovia analyzes historical bus and metro transit data to identify peak delay periods, frequently affected routes, and recurring patterns that impact daily commuters.

---

## The Complete ML Workflow

### 1. Raw Data Collection
The starting point is historical transit data — trip logs, scheduled vs. actual arrival times, route IDs, stop sequences, timestamps, and external factors like weather or events. This data is almost always messy: missing arrival times, duplicate trip records, inconsistent route naming, and outlier values from sensor errors.

**Key principle:** Models only understand numbers. A route name like "Route 42B" or a timestamp like "08:32:00" means nothing to a model until it is transformed into a numerical representation.

---

### 2. Data Cleaning & Preprocessing
Before any modeling, the raw data must be made usable:
- Remove duplicate trip records
- Handle missing delay values (impute or drop)
- Standardize inconsistent formats (e.g., route names, date formats)
- Filter out sensor noise and extreme outliers

In Metrovia, this means ensuring every trip record has a valid scheduled time, actual time, route ID, and stop ID before moving forward.

---

### 3. Feature Engineering
This is the most critical step. Raw columns are transformed into signals the model can actually learn from.

| Raw Data | Engineered Feature | Why It Matters |
|---|---|---|
| Scheduled time & actual time | `delay_minutes` | The target variable |
| Timestamp | `hour_of_day`, `day_of_week`, `is_rush_hour` | Captures peak patterns |
| Date | `is_weekend`, `is_holiday` | Different delay behavior |
| Route ID | `route_avg_delay_30d` | Encodes historical route performance |
| Stop sequence | `stop_position_ratio` | End-of-line stops delay more |
| Weather data | `is_raining`, `temperature_bin` | External delay drivers |

Feature engineering consumes 60–80% of total project effort. A simple model with great features will outperform a complex model with poor features every time.

---

### 4. Model Training
The model learns a function that maps features → predicted delay. For Metrovia, this could be:
- A **regression model** to predict exact delay in minutes
- A **classification model** to predict delay severity (on-time / minor / major)

During training, the model examines thousands of labeled historical trips, measures prediction error using a loss function, and adjusts its internal parameters to minimize that error. The output is a trained model artifact — a saved object that encodes everything learned from the data.

---

### 5. Evaluation (on held-out data)
The model is tested on trips it has never seen before. This gives an honest estimate of real-world performance.

Metrics used:
- **RMSE** — average magnitude of delay prediction error (regression)
- **Precision / Recall** — for classifying severe delays without too many false alarms
- **F1 Score** — balance between catching real delays and avoiding false alerts

Without evaluation on unseen data, there is no way to know if the model actually works or has simply memorized the training set.

---

### 6. Prediction
Once deployed, the model receives new trip data in real time, applies the same feature transformations used during training, and outputs a predicted delay or delay probability.

Example output:
- Route 7, Stop 14, 8:45 AM Monday → **87% probability of delay > 5 minutes**

This is probabilistic, not certain. The model says: "Based on everything learned from historical data, this is the most likely outcome." It does not guarantee the outcome.

**Critical:** The exact same feature transformations applied during training must be applied at prediction time. Any mismatch causes silent, hard-to-debug errors.

---

### 7. Monitoring (Ongoing)
After deployment, the world keeps changing. Bus schedules change, new routes open, commuter patterns shift seasonally. The model's learned assumptions gradually become stale.

Monitoring involves:
- Tracking prediction accuracy over time in production
- Detecting **data drift** — when incoming trip data looks different from training data
- Detecting **concept drift** — when the relationship between features and delays changes
- Triggering retraining when performance drops below acceptable thresholds

A model that was accurate at launch can silently degrade over months without monitoring.

---

## How the Stages Connect

```
Raw Transit Data
      ↓
Data Cleaning (remove noise, fix formats)
      ↓
Feature Engineering (encode time, route history, weather)
      ↓
Model Training (learn delay patterns from labeled trips)
      ↓
Evaluation (test on unseen trips, measure RMSE / F1)
      ↓
Deployment (predict delays on live trip data)
      ↓
Monitoring (track drift, retrain when needed)
```

Each stage feeds the next. A failure at any stage propagates forward — bad data produces bad features, bad features produce a bad model, a bad model produces harmful predictions.

---

## Real-World Example: Predicting Morning Rush Delays on Route 12

**Business problem:** Commuters on Route 12 frequently experience unexpected delays between 8–9 AM. The authority wants to predict these delays 30 minutes in advance.

**Raw data:** Trip logs for Route 12 over 2 years — scheduled departure, actual departure, stop-level timestamps, weather records, special event calendar.

**Features engineered:**
- `delay_last_trip` — did the previous trip on this route run late? (cascading delay signal)
- `is_rush_hour` — binary flag for 7–9 AM and 5–7 PM
- `rain_intensity` — binned rainfall from weather API
- `route_12_avg_delay_7d` — rolling 7-day average delay for this route
- `stops_remaining` — delays compound toward end of route

**Model:** Gradient boosting regressor trained on 18 months of data, evaluated on the most recent 6 months.

**Prediction:** For an upcoming 8:15 AM trip, the model outputs: *predicted delay = 7.3 minutes*. The app alerts commuters to expect a late arrival.

**Monitoring:** After a new bus fleet is introduced, average delays drop. The model starts over-predicting delays. Drift is detected, and the model is retrained on recent data.

---

## Failure Scenario: Data Leakage in Evaluation

**What goes wrong:** During feature engineering, a feature called `actual_arrival_time` is accidentally included in the training data. This column is derived from the same source as the target variable (`delay_minutes`). The model learns to "predict" delays using information it would never have access to at real prediction time.

**Stage of failure:** Evaluation — the model appears to achieve near-perfect accuracy on the test set.

**What happens in production:** The feature `actual_arrival_time` is not available when predicting future trips (since the trip hasn't happened yet), so it is dropped. Model performance collapses immediately after deployment.

**How to diagnose:** Compare features available at training time vs. features available at prediction time. Any feature derived from post-event data is a leakage risk. Always simulate the prediction-time data environment during evaluation.

---

## Key Principles

- Models do not understand business meaning — everything must be encoded as numbers
- Feature engineering determines model success more than algorithm choice
- Always evaluate on data the model has never seen
- Monitoring is not optional — the world changes, and models must adapt
- When a model fails, look at the data and features first, not the algorithm
