
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.anomaly_model import fit
from pipeline.features import extract_features


df = pd.read_csv(ROOT / "datasets" / "sample_data.csv")
normal = df[df["label"] == 0]
model = fit([extract_features(row) for row in normal.to_dict("records")])

correct = 0
for row in df.to_dict("records"):
    prediction = model.predict(extract_features(row))
    correct += int(prediction["is_anomaly"]) == int(row["label"])

print("records:", len(df))
print("accuracy:", correct / len(df))
