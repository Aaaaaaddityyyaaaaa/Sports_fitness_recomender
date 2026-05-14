import pandas as pd

df = pd.read_excel("5000dataset.xlsx")
df.columns = [c.strip() for c in df.columns]

df["tier"], bins = pd.qcut(
    df["Performance_Metric"],
    q=[0, 0.25, 0.50, 0.85, 1.0],
    labels=["poor", "mid", "fit", "super fit"],
    retbins=True,
    duplicates="drop"
)

FEATURES = [
    "Sleep_Hours", "Hydration_Level", "Stress_Level",
    "Recovery_Score", "Training_Hours_Per_Week",
    "Body_Fat_Percentage", "Average_Heart_Rate"
]

for tier in ["poor", "mid", "fit", "super fit"]:
    print(f"\n── {tier} ──")
    subset = df[df["tier"] == tier][FEATURES]
    print(subset.quantile([0.25, 0.5, 0.75]).round(2).to_string())