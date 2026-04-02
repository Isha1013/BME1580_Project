import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, cross_val_predict


FEATURE_FILE = "audio_features.csv"
df = pd.read_csv(FEATURE_FILE)

def run_model(sub_df, label):
    exclude_cols = ["filename", "sound_class", "disease_class", "patient_id"]
    X = sub_df.drop(columns=exclude_cols).values
    y = sub_df["disease_class"].values   

    # Encode labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y_enc)),
        eval_metric="mlogloss",
        use_label_encoder=False,
        max_depth=5,
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate
    y_pred = cross_val_predict(xgb_clf, X_scaled, y_enc, cv=skf)

    print("Classification Report:")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_enc, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"XGBoost Confusion Matrix ({label}) - Sound Classification")
    plt.show()


for prefix in ["B", "D", "E"]:
    subset = df[df["filename"].str.startswith(prefix)]

    run_model(subset, prefix)

run_model(df, "All Filtering Modes")
