import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# 1. Load dataset
# =========================
DATA_PATH = "healthcare-dataset-stroke-data4.csv"
df = pd.read_csv(DATA_PATH)

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# =========================
# 2. Handle missing values
# =========================
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# =========================
# 3. Split features & label
# =========================
X = df.drop("stroke", axis=1)
y = df["stroke"]

# =========================
# 4. Identify column types
# =========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# =========================
# 5. Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# 6. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 7. Define models
# =========================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "Linear Regression": LinearRegression()
}

# =========================
# 8. Training & Evaluation
# =========================
results = []

for model_name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    if model_name == "Linear Regression":
        y_prob = np.clip(clf.predict(X_test), 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

    # ===== ĐỘ TIN CẬY (CHỈ TÍNH KHI MODEL DỰ ĐOÁN STROKE) =====
    stroke_preds = y_pred == 1
    if stroke_preds.sum() > 0:
        confidence = np.mean(y_prob[stroke_preds]) * 100
    else:
        confidence = 0.0

    results.append({
        "Mô hình": model_name,
        "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
        "Precision (%)": precision_score(y_test, y_pred, zero_division=0) * 100,
        "Recall (%)": recall_score(y_test, y_pred) * 100,
        "F1-score (%)": f1_score(y_test, y_pred) * 100,
        "ROC-AUC (%)": roc_auc_score(y_test, y_prob) * 100,
        "Confidence (%)": confidence
    })

# =========================
# 9. Display results
# =========================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="ROC-AUC (%)", ascending=False)
results_df = results_df.round(2)

print("\n===== KẾT QUẢ SO SÁNH MÔ HÌNH (%) =====\n")
print(results_df.to_string(index=False))
