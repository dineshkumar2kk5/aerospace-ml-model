# AeroBeacon — ML-Based Question Balancing Engine

> Intelligent difficulty prediction and balanced exam generation for DGCA aviation exam preparation.

---

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Why Random Forest?](#why-random-forest)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
- [API Reference](#api-reference)
- [Retraining Pipeline](#retraining-pipeline)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AeroBeacon Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐     ┌──────────────────────┐     ┌───────────────────┐  │
│   │          │     │                      │     │                   │  │
│   │  React   │────▶│   Node.js / Express  │────▶│  Flask ML Service │  │
│   │  Client  │     │     (Port 3001)      │     │   (Port 5001)     │  │
│   │          │◀────│                      │◀────│                   │  │
│   └──────────┘     └──────────┬───────────┘     └─────────┬─────────┘  │
│                               │                           │             │
│                               │                           │             │
│                    ┌──────────▼───────────┐    ┌──────────▼──────────┐ │
│                    │                      │    │                     │ │
│                    │   MongoDB Atlas      │    │ difficulty_model.pkl│ │
│                    │   ┌─────────────┐    │    │ label_encoder.pkl   │ │
│                    │   │ Questions   │    │    │ model_metadata.json │ │
│                    │   │ Attempts    │    │    │                     │ │
│                    │   │ ExamSessions│    │    └─────────────────────┘ │
│                    │   └─────────────┘    │                            │
│                    └──────────────────────┘                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                        Data Flow                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Student Request                                                       │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────────────────┐                                               │
│   │ 1. Filter recently  │  (Exclude questions from last 3 exams)       │
│   │    attempted Qs     │                                               │
│   └─────────┬───────────┘                                               │
│             ▼                                                           │
│   ┌─────────────────────┐     ┌──────────────────────┐                  │
│   │ 2. Predict          │────▶│ ML Service           │                  │
│   │    difficulty       │     │ (Random Forest)      │                  │
│   │                     │◀────│                      │                  │
│   └─────────┬───────────┘     └──────────────────────┘                  │
│             │                                                           │
│             │  (Fallback: rule-based heuristics if ML is down)         │
│             ▼                                                           │
│   ┌─────────────────────┐                                               │
│   │ 3. Stratified       │  30% Easy / 50% Medium / 20% Hard           │
│   │    sampling         │                                               │
│   └─────────┬───────────┘                                               │
│             ▼                                                           │
│   ┌─────────────────────┐                                               │
│   │ 4. Fisher-Yates     │  Shuffle Qs + shuffle MCQ options            │
│   │    shuffle          │                                               │
│   └─────────┬───────────┘                                               │
│             ▼                                                           │
│   ┌─────────────────────┐                                               │
│   │ 5. Return balanced  │  With metadata & difficulty breakdown        │
│   │    exam             │                                               │
│   └─────────────────────┘                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Difficulty Prediction (ML)
The system uses a trained **Random Forest classifier** to predict whether a question is **Easy**, **Medium**, or **Hard** based on four features:

| Feature | Description | Source |
|---------|-------------|--------|
| `text_length` | Character count of question text | Computed from `question.text.length` |
| `num_options` | Number of MCQ choices | Computed from `question.options.length` |
| `avg_time_taken` | Average seconds students spend | Aggregated from `Attempt` records |
| `past_accuracy` | Historical correct-answer rate | Aggregated from `Attempt` records |

### 2. Exam Balancing (30/50/20 Rule)
Every generated mock exam targets:
- **30% Easy** — confidence-building warm-up
- **50% Medium** — core competency testing
- **20% Hard** — stretch questions for top performers

### 3. Anti-Repetition
The last **3 exam sessions** for each student are checked. Any question that appeared in those sessions is excluded from the pool.

### 4. Shuffle
Both question order and MCQ option order are randomized using the **Fisher-Yates algorithm**, which guarantees an unbiased, uniform random permutation in O(n) time.

---

## Why Random Forest?

Random Forest was selected as the production model after comparing four classifiers:

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Logistic Regression** | Fast, interpretable | Linear decision boundaries; struggles with non-linear feature interactions |
| **Decision Tree** | Very fast, interpretable | High variance, overfits on small datasets |
| **Random Forest** ✅ | Low variance, handles non-linear patterns, built-in feature importance, robust to outliers | Slightly slower inference than a single tree |
| **Gradient Boosting** | Highest raw accuracy potential | Slower to train, more hyperparameters, overfits without careful tuning |

### Why Random Forest wins for this use case:

1. **Small dataset (500–5000 rows)**: Random Forest's bagging reduces overfitting risk far better than a single Decision Tree, without requiring as much data as Gradient Boosting.

2. **Feature importance**: The `feature_importances_` attribute tells us exactly which question characteristics drive difficulty — critical for explainability in an educational platform.

3. **Robustness**: Random Forest handles noise in student attempt data gracefully (students guess, take breaks, etc.) because it averages across many trees.

4. **No feature scaling needed**: Unlike Logistic Regression, Random Forest doesn't require feature normalization — simplifying the pipeline.

5. **Fast inference**: Prediction is fast enough for real-time exam generation (< 50ms for a 40-question batch on commodity hardware).

6. **Parallelizable**: Both training (`n_jobs=-1`) and inference benefit from multi-core CPUs.

---

## Project Structure

```
feature1/
├── ml/                              # Python ML pipeline
│   ├── generate_dataset.py          # Synthetic dataset generator
│   ├── train_model.py               # Model training & evaluation
│   ├── ml_service.py                # Flask REST API (port 5001)
│   ├── retrain_pipeline.py          # Nightly retraining cron job
│   ├── requirements.txt             # Python dependencies
│   ├── dataset.csv                  # (generated) Training data
│   ├── difficulty_model.pkl         # (generated) Trained model
│   ├── label_encoder.pkl            # (generated) Label encoder
│   ├── model_metadata.json          # (generated) Model metadata
│   ├── model_comparison.png         # (generated) Accuracy chart
│   ├── feature_importance.png       # (generated) Feature chart
│   └── confusion_matrix.png         # (generated) Confusion matrix
│
├── backend/                         # Node.js backend
│   ├── package.json
│   ├── src/
│   │   ├── server.js                # Express app entry point
│   │   ├── models/
│   │   │   ├── Question.js          # Mongoose Question schema
│   │   │   ├── Attempt.js           # Mongoose Attempt schema
│   │   │   └── ExamSession.js       # Mongoose ExamSession schema
│   │   └── services/
│   │       └── mlQuestionService.js # Core balancing engine
│   └── __tests__/
│       └── services/
│           └── mlQuestionService.test.js  # Jest unit tests
│
└── README.md                        # This file
```

---

## Setup Guide

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **MongoDB** (Atlas or local) — optional for initial testing

### 1. Set Up the ML Pipeline

```bash
# Navigate to the ML directory
cd ml

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data
python generate_dataset.py
# Output: dataset.csv (500 rows)

# Train the model
python train_model.py
# Output: difficulty_model.pkl, label_encoder.pkl, model_metadata.json
#         model_comparison.png, feature_importance.png, confusion_matrix.png

# Start the ML microservice
python ml_service.py
# Running on http://0.0.0.0:5001
```

### 2. Verify the ML Service

```bash
# Health check
curl http://localhost:5001/health

# Single prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text_length": 200, "num_options": 4, "avg_time_taken": 65, "past_accuracy": 0.45}'

# Batch prediction
curl -X POST http://localhost:5001/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      {"text_length": 100, "num_options": 4, "avg_time_taken": 30, "past_accuracy": 0.85},
      {"text_length": 300, "num_options": 4, "avg_time_taken": 110, "past_accuracy": 0.2}
    ]
  }'
```

### 3. Set Up the Node.js Backend

```bash
# Navigate to the backend directory
cd backend

# Install dependencies
npm install

# (Optional) Create a .env file
echo "PORT=3001" > .env
echo "MONGO_URI=mongodb://localhost:27017/aerobeacon" >> .env
echo "ML_SERVICE_URL=http://localhost:5001" >> .env

# Start the backend
npm run dev
# Running on http://localhost:3001
```

### 4. Run Tests

```bash
cd backend
npm test
```

---

## API Reference

### Flask ML Service (Port 5001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + model metadata |
| `GET` | `/model-info` | Detailed model metadata |
| `POST` | `/predict` | Predict single question difficulty |
| `POST` | `/predict-batch` | Predict batch of question difficulties |
| `POST` | `/retrain-hook` | Reload model from disk after retraining |

#### `POST /predict`
```json
// Request
{
  "text_length": 180,
  "num_options": 4,
  "avg_time_taken": 55,
  "past_accuracy": 0.62
}

// Response
{
  "difficulty": "Medium",
  "confidence": 0.87,
  "probabilities": {
    "Easy": 0.05,
    "Medium": 0.87,
    "Hard": 0.08
  }
}
```

#### `POST /predict-batch`
```json
// Request
{
  "questions": [
    { "text_length": 100, "num_options": 4, "avg_time_taken": 30, "past_accuracy": 0.85 },
    { "text_length": 300, "num_options": 4, "avg_time_taken": 110, "past_accuracy": 0.20 }
  ]
}

// Response
[
  { "difficulty": "Easy", "confidence": 0.92, "probabilities": {...} },
  { "difficulty": "Hard", "confidence": 0.88, "probabilities": {...} }
]
```

### Node.js Backend (Port 3001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Backend health check |
| `GET` | `/api/ml-status` | ML service connectivity status |
| `POST` | `/api/exam/generate` | Generate a balanced mock exam |

#### `POST /api/exam/generate`
```json
// Request
{
  "studentId": "64f1a2b3c4d5e6f7a8b9c0d1",
  "totalQuestions": 40,
  "topic": "Navigation"  // optional
}

// Response
{
  "questions": [...],
  "metadata": {
    "totalQuestions": 40,
    "difficultyBreakdown": { "easy": 12, "medium": 20, "hard": 8 },
    "predictionSource": "ml_service",
    "generatedAt": "2026-09-30T09:00:00.000Z"
  }
}
```

---

## Retraining Pipeline

The retraining pipeline runs nightly (via cron) to update the model with real student data.

### Cron Schedule (Linux/macOS)
```bash
# Run at 2:00 AM daily
0 2 * * * cd /path/to/ml && /path/to/venv/bin/python retrain_pipeline.py
```

### Windows Task Scheduler
```powershell
# Create a scheduled task
schtasks /create /tn "AeroBeacon Retrain" /tr "python C:\path\to\ml\retrain_pipeline.py" /sc daily /st 02:00
```

### Pipeline Steps
1. **Pull** new student attempt data from MongoDB
2. **Merge** with existing synthetic dataset (real data overwrites synthetic)
3. **Backup** current model (`difficulty_model_old.pkl`)
4. **Retrain** via `train_model.py`
5. **Compare** old vs. new model accuracy on held-out test set
6. **Promote** new model if accuracy improved, or **rollback** if worse
7. **Notify** the running Flask service to reload the model
8. **Log** results to `retrain_log.json` for monitoring

### Safety Guardrails
- Automatic rollback if new model accuracy < old model accuracy
- Automatic rollback if accuracy falls below absolute minimum (70%)
- Console alerts on rollback (Slack/email stubs ready for production)
- All retraining results logged with timestamps

---

## Testing

### Run All Tests
```bash
cd backend
npm test
```

### Test Coverage Report
```bash
npm run test:coverage
```

### Test Categories

| Category | Tests | What's Verified |
|----------|-------|-----------------|
| Fisher-Yates Shuffle | 5 | Immutability, correctness, edge cases, statistical uniformity |
| Option Shuffling | 4 | Correct answer preservation, validation |
| Recent Question Filter | 5 | History exclusion, session lookback, empty-pool safety |
| Stratified Sampling | 5 | Distribution ratios, shortfall handling, edge cases |
| Rule-Based Fallback | 6 | Heuristic accuracy, confidence bounds, missing fields |
| Balanced Exam Builder | 6 | End-to-end pipeline, metadata, filtering, shuffle |

---

## Troubleshooting

### ML Service Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `Model file not found` | Model not trained yet | Run `python generate_dataset.py && python train_model.py` |
| `Port 5001 already in use` | Another process on the port | Kill it: `lsof -ti:5001 \| xargs kill` (Unix) or use `--port 5050` |
| `Rate limit exceeded (429)` | Too many requests/sec | Wait 1 second, or increase `max_tokens` in `ml_service.py` |
| Low prediction confidence | Features don't distinguish well | Check if `past_accuracy` and `avg_time_taken` have sufficient variance |

### Node.js Backend Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `ML service unavailable` | Flask not running | Start `python ml_service.py`. Backend will use rule-based fallback automatically |
| `No questions in pool` | Empty database | Seed MongoDB with questions, or add JSON fallback data |
| `MongoDB connection failed` | URI incorrect or DB down | Check `MONGO_URI` in `.env`. Backend will warn but continue |
| Exam has fewer questions than requested | Pool too small after filtering | Add more questions or reduce `lastN` recent-session lookback |

### Retraining Pipeline Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `pymongo not installed` | Missing dependency | `pip install pymongo` |
| Model rolled back repeatedly | Real data contradicts synthetic | Increase real data volume; synthetic labels may need correction |
| `train_model.py failed` | Dataset corruption or missing columns | Validate `dataset_real.csv` schema matches expected columns |

### General Tips

1. **Check ML service health first**: `curl http://localhost:5001/health`
2. **Check backend health**: `curl http://localhost:3001/api/health`
3. **Check ML connectivity from backend**: `curl http://localhost:3001/api/ml-status`
4. **View retrain history**: `cat ml/retrain_log.json`
5. **Model metadata**: `cat ml/model_metadata.json`
6. **Run tests before deploying**: `cd backend && npm test`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3001` | Node.js backend port |
| `MONGO_URI` | `mongodb://localhost:27017/aerobeacon` | MongoDB connection string |
| `ML_SERVICE_URL` | `http://localhost:5001` | Flask ML service URL |
| `DEBUG` | `false` | Enable debug logging in Node.js |

---

## License

Internal — AeroBeacon DGCA Exam Preparation Platform.
