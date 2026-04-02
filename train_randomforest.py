import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict

FEATURE_FILE = "audio_features.csv"
df = pd.read_csv(FEATURE_FILE)

def run_model(sub_df, label):
    # exclude non-numeric columns for ML
    exclude_cols = ["filename", "sound_class", "disease_class", "patient_id"]
    X = sub_df.drop(columns=exclude_cols).values
    y = sub_df["disease_class"].values

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate
    y_pred = cross_val_predict(rf, X_scaled, y, cv=skf)

    print("Classification Report:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y), cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Random Forest Confusion Matrix ({label}) - Sound Classification")
    plt.show()

for prefix in ["B", "D", "E"]:
    subset = df[df["filename"].str.startswith(prefix)]

    run_model(subset, prefix)

run_model(df, "All Filtering Modes")
