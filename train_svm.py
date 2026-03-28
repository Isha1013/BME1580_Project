import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


FEATURE_FILE = "audio_features.csv"
df = pd.read_csv(FEATURE_FILE)

def run_model(sub_df, label):
    # exclude non-numeric columns for ML
    exclude_cols = ["filename", "sound_class", "disease_class", "patient_id"]
    X = sub_df.drop(columns=exclude_cols).values
    y = sub_df["sound_class"].values
    patient_ids = sub_df["patient_id"].values

    # Patient-level split
    # ensure same patient doesn't appear in train and test
    unique_patients = sub_df["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )

    train_idx = sub_df["patient_id"].isin(train_patients)
    test_idx = sub_df["patient_id"].isin(test_patients)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    svm_clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42)
    svm_clf.fit(X_train, y_train)

    # Evaluate
    y_pred = svm_clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=svm_clf.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=svm_clf.classes_, yticklabels=svm_clf.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"SVM Confusion Matrix ({label}) - Sound Classification")
    plt.show()

for prefix in ["B", "D", "E"]:
    subset = df[df["filename"].str.startswith(prefix)]

    run_model(subset, prefix)

run_model(df, "All Filtering Modes")
