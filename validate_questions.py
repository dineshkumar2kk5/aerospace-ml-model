"""
validate_questions.py — Question Bank Integrity & Schema Validator
===================================================================
Run this script to verify that backend/data/questions.json adheres to all
DGCA AeroBeacon standards, valid topics, and proper data types.
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

def validate(filepath="backend/data/questions.json"):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        try:
            questions = json.load(f)
        except Exception as e:
            print(f"Error: JSON parsing failed: {e}")
            sys.exit(1)

    print(f"Loaded {len(questions)} questions from {filepath} for validation...")
    errors = []
    topic_counts = {}
    diff_counts = {"Easy": 0, "Medium": 0, "Hard": 0}

    for idx, q in enumerate(questions):
        qid = q.get("id", f"Index_{idx}")

        # Check required fields
        for field in ["id", "text", "topic", "options", "correctIndex", "difficulty", "avgTimeTaken", "pastAccuracy"]:
            if field not in q:
                errors.append(f"[{qid}] Missing required field '{field}'")

        # Validate topic
        topic = q.get("topic")
        if topic not in VALID_TOPICS:
            errors.append(f"[{qid}] Invalid topic: '{topic}'")
        else:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Validate options & correctIndex
        opts = q.get("options", [])
        if not isinstance(opts, list) or len(opts) < 2:
            errors.append(f"[{qid}] 'options' must be a list with at least 2 choices")
        else:
            c_idx = q.get("correctIndex")
            if not isinstance(c_idx, int) or c_idx < 0 or c_idx >= len(opts):
                errors.append(f"[{qid}] 'correctIndex' ({c_idx}) is out of range for {len(opts)} options")

        # Validate difficulty
        diff = q.get("difficulty")
        if diff not in ["Easy", "Medium", "Hard"]:
            errors.append(f"[{qid}] Invalid difficulty: '{diff}'")
        else:
            diff_counts[diff] = diff_counts.get(diff, 0) + 1

        # Validate pastAccuracy (0.0 to 1.0)
        acc = q.get("pastAccuracy", 0)
        if not isinstance(acc, (int, float)) or not (0.0 <= acc <= 1.0):
            errors.append(f"[{qid}] 'pastAccuracy' ({acc}) must be a number between 0.0 and 1.0")

        # Validate avgTimeTaken (10s to 300s)
        time_taken = q.get("avgTimeTaken", 0)
        if not isinstance(time_taken, (int, float)) or not (10 <= time_taken <= 300):
            errors.append(f"[{qid}] 'avgTimeTaken' ({time_taken}) must be between 10 and 300 seconds")

    if errors:
        print(f"\n❌ Validation FAILED with {len(errors)} issues:")
        for err in errors[:20]:
            print(f"  • {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more issues.")
        sys.exit(1)
    else:
        print("\n✅ VALIDATION PASSED: All questions are structurally sound and compliant!")
        print(f"\nTotal Questions: {len(questions)}")
        print(f"Total Topics: {len(topic_counts)} / 28 covered")
        print("\nDifficulty Distribution:")
        for d in ["Easy", "Medium", "Hard"]:
            cnt = diff_counts[d]
            pct = (cnt / len(questions)) * 100
            print(f"  • {d:6s}: {cnt:4d} ({pct:5.1f}%)")

if __name__ == "__main__":
    validate()
