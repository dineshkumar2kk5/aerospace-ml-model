# AeroBeacon — Adding Multiple Questions Prompt & Ingestion Guide
### Master AI Prompt Templates, Schema Guidelines & Ingestion Workflows

This guide provides instructions and copy-paste AI prompt templates for instructors, content managers, and developers to convert raw aviation textbook pages, PDF scans, or notes into AeroBeacon-ready questions.

---

## 1. Master AI Ingestion Prompt

Copy and paste the entire block below into any AI model (ChatGPT, Claude, Gemini, or local LLM) along with your raw textbook pages, PDF OCR text, or practice tests:

````markdown
You are an expert DGCA Aviation Ground Instructor and Senior Question Bank Developer for AeroBeacon (India DGCA CPL/ATPL Pilot Examination System).

Your task is to extract, clean, and convert the provided aviation text or practice questions into a valid JSON array adhering strictly to the AeroBeacon question schema.

### MANDATORY RULES:
1. TOPIC CLASSIFICATION:
   Every question must strictly use one of the official 28 DGCA IC Joshi Aviation Meteorology chapters:
   - "Atmosphere"
   - "Atmospheric Pressure"
   - "Temperature"
   - "Air Density"
   - "Humidity"
   - "Winds"
   - "Visibility and Fog"
   - "Vertical Motion and Clouds"
   - "Stability and Instability"
   - "Optical Phenomena"
   - "Precipitation"
   - "Ice Accretion"
   - "Thunderstorm"
   - "Airmasses Fronts and Western Disturbances"
   - "Jet Streams"
   - "Clear Air Turbulence"
   - "Tropical Systems"
   - "Climatology of India"
   - "General Circulation"
   - "Meteorological Services for Aviation"
   - "Aviation Weather Reports (METAR and SPECI)"
   - "Aerodrome Forecasts (TAF and TREND)"
   - "SIGMET and AIRMET Warnings"
   - "World Area Forecast System (WAFS and SIGWX)"
   - "Radar Meteorology"
   - "Satellite Meteorology"
   - "Altimetry and Pressure Settings"
   - "Flight Weather Planning and Route Hazards"

2. DIFFICULTY CALIBRATION & REALISTIC METRICS:
   For each question, classify its difficulty and assign realistic student behavioral metrics:
   - "Easy": Direct definition, standard recall, or foundational concept.
     * avgTimeTaken: integer between 25 and 42 (seconds).
     * pastAccuracy: float between 0.80 and 0.95 (representing 80% to 95% pass rate).
   - "Medium": Standard scenario, METAR decoding, cloud interpretation, or standard rule application.
     * avgTimeTaken: integer between 45 and 75 (seconds).
     * pastAccuracy: float between 0.55 and 0.78 (representing 55% to 78% pass rate).
   - "Hard": Multi-step calculation (altimetry error, ISA deviation formula), edge cases, complex hazards.
     * avgTimeTaken: integer between 80 and 130 (seconds).
     * pastAccuracy: float between 0.20 and 0.48 (representing 20% to 48% pass rate).

3. MCQ OPTIONS FORMAT:
   - "options": An array of strings containing 3 or 4 answer choices (e.g. ["Choice A", "Choice B", "Choice C", "Choice D"]).
   - "correctIndex": Integer (0 for 1st option, 1 for 2nd option, 2 for 3rd option, 3 for 4th option).
   - "explanation": Concise, authoritative explanation citing standard meteorological principles or ICAO Annex 3.

4. OUTPUT FORMAT:
   - Return ONLY a raw JSON array of objects.
   - Do NOT wrap in markdown explanation or conversational text. Output pure JSON.

### SCHEMA REFERENCE:
```json
[
  {
    "id": "joshi-XXX",
    "text": "The lowest layer of the atmosphere containing 75% of atmospheric mass is:",
    "topic": "Atmosphere",
    "subtopic": "Atmosphere",
    "options": [
      "Troposphere",
      "Stratosphere",
      "Mesosphere"
    ],
    "correctIndex": 0,
    "difficulty": "Easy",
    "avgTimeTaken": 32,
    "pastAccuracy": 0.88,
    "explanation": "Troposphere extends from the surface to the tropopause and contains roughly 75% of atmospheric mass and 99% of water vapor."
  }
]
```

---
RAW AVIATION TEXT / QUESTIONS TO PROCESS:
[PASTE YOUR RAW BOOK TEXT, PDF OCR, OR QUESTIONS HERE]
````

---

## 2. Ingestion Workflow: Adding the Questions to AeroBeacon

Once you have generated the JSON array using the prompt above, follow these steps to integrate the new questions into the live system:

### Step 1: Append Questions to `backend/data/questions.json`
Open [`backend/data/questions.json`](file:///c:/Users/dinesh/OneDrive/Desktop/feature1/backend/data/questions.json) and add the new question objects to the array. Ensure valid JSON syntax (commas separating objects).

### Step 2: Sync to `ml/ic_joshi_questions.json`
Also append the question objects to [`ml/ic_joshi_questions.json`](file:///c:/Users/dinesh/OneDrive/Desktop/feature1/ml/ic_joshi_questions.json).

### Step 3: Regenerate Training Dataset (`dataset.csv`)
Run the dataset synchronization script to update `ml/dataset.csv`:
```bash
python -c "
import json, pandas as pd
with open('backend/data/questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

rows = []
for q in questions:
    rows.append({
        'question_id': q['id'],
        'topic': q['topic'],
        'subtopic': q.get('subtopic', q['topic']),
        'text_length': len(q['text']),
        'num_options': len(q['options']),
        'avg_time_taken': q['avgTimeTaken'],
        'past_accuracy': q['pastAccuracy'],
        'difficulty': q['difficulty']
    })

df = pd.DataFrame(rows)
df.to_csv('ml/dataset.csv', index=False)
print(f'Synchronized {len(df)} questions to ml/dataset.csv')
"
```

### Step 4: Retrain the Local Machine Learning Model
Run the model training pipeline:
```bash
cd ml
python train_model.py
```
This performs:
1. 5-Fold Stratified Cross-Validation across all updated questions.
2. GridSearchCV hyperparameter tuning on the `RandomForestClassifier`.
3. Overwrites `difficulty_model.pkl`, `label_encoder.pkl`, and `model_metadata.json`.
4. Updates visualization charts (`confusion_matrix.png`, `feature_importance.png`, `model_comparison.png`).

### Step 5: Hot-Reload the Live Microservice
Send a zero-downtime reload trigger to the running Flask service:
```bash
python -c "
import urllib.request, json
req = urllib.request.Request('http://127.0.0.1:5001/retrain-hook', data=b'{}', headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(req) as resp:
    print('Hot Reload:', json.loads(resp.read()))
"
```
The active model weights will update in memory within milliseconds with **zero downtime**.

---

## 3. Automated Validation Script

Before deploying new questions, run this validation script to ensure there are no schema violations, out-of-range values, or missing fields:

```python
# validate_questions.py
import json
from pathlib import Path

VALID_TOPICS = {
    "Atmosphere", "Atmospheric Pressure", "Temperature", "Air Density",
    "Humidity", "Winds", "Visibility and Fog", "Vertical Motion and Clouds",
    "Stability and Instability", "Optical Phenomena", "Precipitation",
    "Ice Accretion", "Thunderstorm", "Airmasses Fronts and Western Disturbances",
    "Jet Streams", "Clear Air Turbulence", "Tropical Systems", "Climatology of India",
    "General Circulation", "Meteorological Services for Aviation",
    "Aviation Weather Reports (METAR and SPECI)", "Aerodrome Forecasts (TAF and TREND)",
    "SIGMET and AIRMET Warnings", "World Area Forecast System (WAFS and SIGWX)",
    "Radar Meteorology", "Satellite Meteorology", "Altimetry and Pressure Settings",
    "Flight Weather Planning and Route Hazards"
}

def validate_pool(filepath="backend/data/questions.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)

    errors = []
    print(f"Validating {len(questions)} questions from {filepath}...")

    for idx, q in enumerate(questions):
        qid = q.get("id", f"Index {idx}")

        # Check required fields
        for field in ["id", "text", "topic", "options", "correctIndex", "difficulty", "avgTimeTaken", "pastAccuracy"]:
            if field not in q:
                errors.append(f"[{qid}] Missing required field '{field}'")

        # Validate topic
        if q.get("topic") not in VALID_TOPICS:
            errors.append(f"[{qid}] Invalid topic: '{q.get('topic')}'")

        # Validate options & correctIndex
        opts = q.get("options", [])
        if not isinstance(opts, list) or len(opts) < 2:
            errors.append(f"[{qid}] 'options' must be a list with at least 2 choices")
        else:
            c_idx = q.get("correctIndex")
            if not isinstance(c_idx, int) or c_idx < 0 or c_idx >= len(opts):
                errors.append(f"[{qid}] 'correctIndex' ({c_idx}) is out of range for {len(opts)} options")

        # Validate difficulty
        if q.get("difficulty") not in ["Easy", "Medium", "Hard"]:
            errors.append(f"[{qid}] Invalid difficulty: '{q.get('difficulty')}'")

        # Validate pastAccuracy (0.0 to 1.0)
        acc = q.get("pastAccuracy", 0)
        if not (0.0 <= acc <= 1.0):
            errors.append(f"[{qid}] 'pastAccuracy' ({acc}) must be between 0.0 and 1.0")

        # Validate avgTimeTaken (10s to 300s)
        time_taken = q.get("avgTimeTaken", 0)
        if not (10 <= time_taken <= 300):
            errors.append(f"[{qid}] 'avgTimeTaken' ({time_taken}) must be between 10 and 300 seconds")

    if errors:
        print(f"❌ Found {len(errors)} validation errors:")
        for err in errors[:15]:
            print(f"  - {err}")
        if len(errors) > 15:
            print(f"  ... and {len(errors) - 15} more.")
        return False
    else:
        print("✅ All questions passed schema validation successfully!")
        return True

if __name__ == "__main__":
    validate_pool()
```

Save and run:
```bash
python validate_questions.py
```
Once validated, your question bank will seamlessly integrate into the live exam balancer.
