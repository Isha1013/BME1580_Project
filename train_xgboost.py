import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

FEATURE_FILE = "audio_features.csv"
df = pd.read_csv(FEATURE_FILE)

exclude_cols = ["filename", "sound_class", "disease_class", "patient_id"]
X = df.drop(columns=exclude_cols).values
y = df["disease_class"].values   
patient_ids = df["patient_id"].values

# Encode labels to integers
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Patient-level split
unique_patients = np.unique(patient_ids)
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

train_idx = np.isin(patient_ids, train_patients)
test_idx = np.isin(patient_ids, test_patients)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_enc[train_idx], y_enc[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)


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

xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)


# Evaluate model
y_pred = xgb_clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("XGBoost Confusion Matrix - Disease Type")
plt.show()


# Feature importance
xgb.plot_importance(xgb_clf, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()