from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

scaler = joblib.load("model/artifact/scaler.pkl")
model  = joblib.load("model/artifact/my_best_model.pkl")


# ─────────────────────────────────────────────
# PYDANTIC MODEL
# ─────────────────────────────────────────────

class AthleteInput(BaseModel):
    Age:                     float
    Height_cm:               float
    Weight_kg:               float
    BMI:                     float
    Training_Hours_Per_Week: float
    Training_Intensity:      float
    Sleep_Hours:             float
    Hydration_Level:         float
    VO2_Max:                 float
    Average_Heart_Rate:      float
    Resting_Heart_Rate:      float
    Mental_Focus_Level:      float
    Stress_Level:            float
    Motivation_Level:        float
    Body_Fat_Percentage:     float
    Muscle_Mass:             float
    Daily_Caloric_Intake:    float
    Protein_Intake:          float
    Recovery_Score:          float
    Reaction_Time_ms:        float
    Speed_Score:             float
    Endurance_Score:         float
    Flexibility_Score:       float
    Experience_Years:        float
    Injury_History:          int


# ─────────────────────────────────────────────
# FEATURE ORDER
# ─────────────────────────────────────────────

FEATURE_ORDER = [
    "Age", "Height_cm", "Weight_kg", "BMI",
    "Training_Hours_Per_Week", "Training_Intensity",
    "Sleep_Hours", "Hydration_Level", "VO2_Max",
    "Average_Heart_Rate", "Resting_Heart_Rate",
    "Mental_Focus_Level", "Stress_Level", "Motivation_Level",
    "Body_Fat_Percentage", "Muscle_Mass", "Daily_Caloric_Intake",
    "Protein_Intake", "Recovery_Score", "Reaction_Time_ms",
    "Speed_Score", "Endurance_Score", "Flexibility_Score",
    "Experience_Years", "Injury_History"
]


# ─────────────────────────────────────────────
# TIER ASSIGNMENT
# ─────────────────────────────────────────────

TIER_BINS   = [0.0, 40.0, 55.0, 70.0, 100.0]  # replace with your actual qcut bin edges
TIER_LABELS = ["poor", "mid", "fit", "super fit"]
TIER_ORDER  = ["poor", "mid", "fit", "super fit"]


def assign_tier(score: float) -> str:
    for i, (lo, hi) in enumerate(zip(TIER_BINS, TIER_BINS[1:])):
        if lo <= score <= hi:
            return TIER_LABELS[i]
    return TIER_LABELS[-1]


# ─────────────────────────────────────────────
# IQR BOUNDARIES (from your qcut output)
# ─────────────────────────────────────────────

BOUNDARIES = {
    "Sleep_Hours": {
        "poor":      {"q25": 5.81, "q50": 6.68, "q75": 7.67},
        "mid":       {"q25": 5.98, "q50": 6.93, "q75": 7.87},
        "fit":       {"q25": 6.08, "q50": 7.03, "q75": 8.01},
        "super fit": {"q25": 6.18, "q50": 7.24, "q75": 8.14},
    },
    "Hydration_Level": {
        "poor":      {"q25": 2.32, "q50": 3.19, "q75": 4.02},
        "mid":       {"q25": 2.28, "q50": 3.20, "q75": 4.13},
        "fit":       {"q25": 2.39, "q50": 3.27, "q75": 4.12},
        "super fit": {"q25": 2.52, "q50": 3.42, "q75": 4.27},
    },
    "Stress_Level": {
        "poor":      {"q25": 4.18, "q50": 6.47, "q75": 8.27},
        "mid":       {"q25": 3.55, "q50": 5.74, "q75": 8.11},
        "fit":       {"q25": 3.02, "q50": 5.25, "q75": 7.46},
        "super fit": {"q25": 2.64, "q50": 4.30, "q75": 6.65},
    },
    "Recovery_Score": {
        "poor":      {"q25": 48.22, "q50": 59.66, "q75": 74.78},
        "mid":       {"q25": 52.80, "q50": 67.26, "q75": 81.92},
        "fit":       {"q25": 58.80, "q50": 73.89, "q75": 87.80},
        "super fit": {"q25": 66.15, "q50": 80.73, "q75": 91.70},
    },
    "Training_Hours_Per_Week": {
        "poor":      {"q25":  8.38, "q50": 12.54, "q75": 17.61},
        "mid":       {"q25": 10.05, "q50": 14.62, "q75": 19.63},
        "fit":       {"q25": 11.28, "q50": 16.16, "q75": 20.66},
        "super fit": {"q25": 13.76, "q50": 17.86, "q75": 22.01},
    },
    "Body_Fat_Percentage": {
        "poor":      {"q25": 15.55, "q50": 20.62, "q75": 24.53},
        "mid":       {"q25": 14.12, "q50": 18.93, "q75": 23.48},
        "fit":       {"q25": 12.23, "q50": 16.94, "q75": 21.99},
        "super fit": {"q25": 11.10, "q50": 15.28, "q75": 20.19},
    },
    "Average_Heart_Rate": {
        "poor":      {"q25": 65.11, "q50": 75.94, "q75": 85.91},
        "mid":       {"q25": 64.30, "q50": 74.37, "q75": 84.96},
        "fit":       {"q25": 65.50, "q50": 74.82, "q75": 84.96},
        "super fit": {"q25": 63.42, "q50": 72.97, "q75": 84.24},
    },
}

HIGHER_IS_BETTER = {
    "Sleep_Hours":             True,
    "Hydration_Level":         True,
    "Stress_Level":            False,
    "Recovery_Score":          True,
    "Training_Hours_Per_Week": True,
    "Body_Fat_Percentage":     False,
    "Average_Heart_Rate":      False,
}


# ─────────────────────────────────────────────
# PRESCRIPTION ENGINE
# ─────────────────────────────────────────────

def classify_feature(value: float, feature: str, current_tier: str):
    b    = BOUNDARIES[feature][current_tier]
    high = HIGHER_IS_BETTER[feature]
    if high:
        if value < b["q25"]:
            return "lagging", b["q50"]
        elif value < b["q50"]:
            return "below_median", b["q75"]
        else:
            return "good", None
    else:
        if value > b["q75"]:
            return "lagging", b["q50"]
        elif value > b["q50"]:
            return "below_median", b["q25"]
        else:
            return "good", None


def next_tier_target(feature: str, current_tier: str):
    idx = TIER_ORDER.index(current_tier)
    if idx == len(TIER_ORDER) - 1:
        return None, None
    next_tier = TIER_ORDER[idx + 1]
    b    = BOUNDARIES[feature][next_tier]
    high = HIGHER_IS_BETTER[feature]
    return next_tier, b["q25"] if high else b["q75"]


def generate_prescription(athlete: dict, current_tier: str) -> str:
    lagging      = []
    below_median = []
    good         = []

    for feature, high in HIGHER_IS_BETTER.items():
        value = athlete.get(feature)
        if value is None:
            continue

        status, target         = classify_feature(value, feature, current_tier)
        next_tier, next_target = next_tier_target(feature, current_tier)

        entry = {
            "feature":     feature,
            "value":       value,
            "status":      status,
            "direction":   "Increase" if high else "Decrease",
            "target":      target,
            "next_tier":   next_tier,
            "next_target": next_target,
            "sign":        ">=" if high else "<=",
        }

        if status == "lagging":
            lagging.append(entry)
        elif status == "below_median":
            below_median.append(entry)
        else:
            good.append(entry)

    lines = []
    lines.append(f"## AI Sports Prescription")
    lines.append(f"**Current Tier:** {current_tier.upper()}\n")

    if lagging:
        lines.append("### ⚠️ Priority Improvements (Significantly Lagging)")
        for e in lagging:
            lines.append(f"- **{e['feature']}**: Current value `{e['value']}`")
            lines.append(f"  - {e['direction']} to `{e['sign']}{e['target']}` *(median of {current_tier} tier)*")
            if e["next_tier"]:
                lines.append(f"  - To reach **{e['next_tier']}**: target `{e['sign']}{e['next_target']}`")

    if below_median:
        lines.append("\n### 📈 Room for Improvement (Below Tier Median)")
        for e in below_median:
            lines.append(f"- **{e['feature']}**: Current value `{e['value']}`")
            lines.append(f"  - {e['direction']} to `{e['sign']}{e['target']}`")
            if e["next_tier"]:
                lines.append(f"  - To reach **{e['next_tier']}**: target `{e['sign']}{e['next_target']}`")

    if good:
        lines.append("\n### ✅ Strengths — Keep Maintaining")
        for f in good:
            lines.append(f"- **{f}**")

    if current_tier == "super fit":
        lines.append("\n### 🏆 Already at Top Tier")
        lines.append("Focus on maintaining all metrics consistently.")

    return "\n".join(lines)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.post("/predict")
def predict(data: AthleteInput):
    df          = pd.DataFrame([data.dict()])[FEATURE_ORDER]
    transformed = scaler.transform(df)
    score       = float(model.predict(transformed)[0])
    tier        = assign_tier(score)

    athlete_features = {
        "Sleep_Hours":             data.Sleep_Hours,
        "Hydration_Level":         data.Hydration_Level,
        "Stress_Level":            data.Stress_Level,
        "Recovery_Score":          data.Recovery_Score,
        "Training_Hours_Per_Week": data.Training_Hours_Per_Week,
        "Body_Fat_Percentage":     data.Body_Fat_Percentage,
        "Average_Heart_Rate":      data.Average_Heart_Rate,
    }

    prescription = generate_prescription(athlete_features, tier)

    return {
        "performance_metric": round(score, 3),
        "tier":               tier,
        "prescription":       prescription,
    }
