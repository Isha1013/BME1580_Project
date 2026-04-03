import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score



# --- Parameters ---
FEATURE_FILE = "audio_features.csv"
TARGET_COLUMN = "sound_class"
FILTER_PREFIXES = ["B", "D", "E"]  
N_SPLITS = 5
RANDOM_STATE = 42

# --- Load features ---
df = pd.read_csv(FEATURE_FILE)

def prepare_features(sub_df, binary=True):
    """
    Prepare features and labels for ML.

    Parameters:
        sub_df (pd.DataFrame): subset of data
        binary (bool): if True, combines heart & lung disease into 'Disease' class

    Returns:
        X (pd.DataFrame): feature matrix
        y_encoded (np.array): encoded target
        le (LabelEncoder): fitted label encoder
        sample_weights (np.array): class-balanced sample weights
    """
    sub_df = sub_df.dropna(subset=["patient_id"])

    # if binary classification
    if binary:
        sub_df[TARGET_COLUMN] = sub_df[TARGET_COLUMN].replace(
            {"Heart disease": "Disease", "Lung disease": "Disease"}
        )

    # Drop non-feature columns
    X = sub_df.drop(columns=["filename", "sound_class", "disease_class", "patient_id"], errors="ignore")

    # Encode gender
    if "gender" in X.columns:
        X["gender"] = X["gender"].map({"Male": 0, "Female": 1})
        X["gender"] = X["gender"].fillna(-1)

    # Fill missing age
    if "age" in X.columns:
        X["age"] = X["age"].fillna(X["age"].median())

    # Ensure numeric
    X = X.astype(float)

    # Encode target
    y = sub_df[TARGET_COLUMN]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Sample weights for class imbalance
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_encoded)

    return X, y_encoded, le, sample_weights

def run_knn(sub_df, label):
    X, y, le, sample_weights = prepare_features(sub_df, binary=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_f1_scores = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n--- Fold {fold+1} ({label}) ---")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        weights_train = sample_weights[train_idx]

        model = KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        )

        # KNN does NOT support sample_weight in fit → ignore weights
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred, average="weighted")
        fold_f1_scores.append(f1)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        print(f"F1 Score (Fold {fold+1}): {f1:.4f}")

    print(f"\nAverage F1 Score ({label}): {np.mean(fold_f1_scores):.4f}")

    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))

    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"KNN Confusion Matrix ({label})")
    plt.show()


# --- Run per filter mode ---
for prefix in FILTER_PREFIXES:
    subset = df[df["filename"].str.startswith(prefix)]
    run_knn(subset, f"Filter {prefix}")

# --- Run on all data ---
run_knn(df, "All Filters")
