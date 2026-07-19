import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import joblib

# -------------------------------------------------
# 1. Paths & data loading
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "Data" / "JNN_DS_with_binary_features_merged.csv"
print(data_path)
df = pd.read_csv(data_path)

# Remove non-positive RT if any (safety)
df = df[df["avg_response_time_ms"] > 0].copy()

# -------------------------------------------------
# 2. Define features & target
# -------------------------------------------------
# Binary service columns (from the merged dataset)
service_cols = [
    "adv1", "adv2",
    "analytics1", "analytics2",
    "breaking",
    "content1", "content2",

    "media1", "media2",
    "recommendation1", "recommendation2",
]

# Workload columns
num_cols = [ "actual_rpm"]

# Target: log-transformed RT
df["log_rt"] = np.log1p(df["avg_response_time_ms"])

X = df[service_cols + num_cols]
y = df["log_rt"]

# -------------------------------------------------
# 3. Train/validation split (stratified by RPM)
# -------------------------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=pd.cut(df["actual_rpm"], bins=5)
)

# -------------------------------------------------
# 4. Preprocessing:
#    - pass-through binary service flags
#    - scale workload columns
# -------------------------------------------------
preproc = ColumnTransformer(
    transformers=[
        ("bin", "passthrough", service_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
)

preproc.fit(X_train)

X_train_t = preproc.transform(X_train)
X_valid_t = preproc.transform(X_valid)

# -------------------------------------------------
# 5. Monotone constraints:
#    binaries: 0 (no constraint)
#    num_users & actual_rpm: +1 (RT should not decrease when load increases)
# -------------------------------------------------
n_bin = len(service_cols)
n_num = len(num_cols)

constraints = [0] * n_bin +  [1, 0] #[1] * n_num
constraints_str = "(" + ",".join(map(str, constraints)) + ")"

# -------------------------------------------------
# 6. XGBoost + hyperparameter search
# -------------------------------------------------
xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    motone_constraints=constraints_str,
)

param_dist = {
    "n_estimators": randint(150, 500),
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.01, 0.2),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train_t, y_train)

print("Best params:", search.best_params_)
print("Best CV score (neg RMSE, log-space):", search.best_score_)

best = search.best_estimator_

# Optional: refine with early stopping on valid set
best.fit(
    X_train_t,
    y_train,
    eval_set=[(X_valid_t, y_valid)],
    verbose=False,
)

# -------------------------------------------------
# 7. Evaluate in ms-space
# -------------------------------------------------
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 7. Evaluate in ms-space
pred_log = best.predict(X_valid_t)

# 1) RMSE in log-space (optional sanity check)
rmse_log = np.sqrt(mean_squared_error(y_valid, pred_log))
print(f"RMSE (log-space): {rmse_log:.4f}")

# 2) Convert back to milliseconds
true_ms = np.expm1(y_valid.values)   # y_valid is log_rt
pred_ms = np.expm1(pred_log)

# MSE -> RMSE
mse = mean_squared_error(true_ms, pred_ms)
rmse = np.sqrt(mse)

r2 = r2_score(true_ms, pred_ms)

print(f"Final RMSE (ms): {rmse:.2f}")
print(f"Final R²: {r2:.4f}")

# -------------------------------------------------
# 8. Save model, preprocessor, and feature metadata
# -------------------------------------------------
models_dir = BASE_DIR / "Models"
models_dir.mkdir(exist_ok=True)

joblib.dump(best, models_dir / "JNN_RT_XGB_binary.pkl")
joblib.dump(preproc, models_dir / "JNN_RT_preproc_binary.pkl")

meta = {
    "service_cols": service_cols,
    "num_cols": num_cols,
}
joblib.dump(meta, models_dir / "JNN_RT_meta_binary.pkl")

print("Saved JNN RT model (binary features) and metadata.")
