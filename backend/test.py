# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error, mean_squared_error


# class GlobalElectricityModelDebugFixedV2:
#     """
#     GLOBAL ELECTRICITY MODEL — STABLE VERSION

#     FIX CRITICAL:
#     - robust scaling per client (median/IQR)
#     - clip extreme spikes (99.5%)
#     - keep log option
#     - reduce RMSE explosion
#     - full debug logs
#     """

#     def __init__(self, dataset_path, test_size=0.2, use_log=True):
#         self.dataset_path = dataset_path
#         self.test_size = test_size
#         self.use_log = use_log

#         self.raw_df = None
#         self.df = None

#         self.model = lgb.LGBMRegressor(
#             n_estimators=1200,
#             learning_rate=0.02,
#             num_leaves=64,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )

#         self.features = [
#             "lag_1", "lag_24", "lag_48", "lag_168",
#             "rolling_24",
#             "hour", "day_of_week", "month",
#             "client_code",
#             "is_zero"
#         ]

#     # ======================
#     # 1. LOAD
#     # ======================
#     def load_data(self):
#         print("\n>>> [1/6] Loading data...")

#         df = pd.read_csv(
#             self.dataset_path,
#             sep=";",
#             decimal=",",
#             parse_dates=[0]
#         )

#         df = df.rename(columns={"Unnamed: 0": "time"})
#         df = df.set_index("time")

#         self.raw_df = df

#         print(f"[LOAD] Shape: {df.shape}")
#         print(f"[LOAD] Time range: {df.index.min()} → {df.index.max()}")
#         print(f"[LOAD] Clients: {df.shape[1]}")
#         print(f"[LOAD] Zero %: {(df == 0).sum().sum() / df.size:.2%}")

#         return df

#     # ======================
#     # 2. LONG FORMAT
#     # ======================
#     def to_long(self):
#         print("\n>>> [2/6] Wide -> Long...")

#         df = self.raw_df.reset_index().melt(
#             id_vars="time",
#             var_name="client_id",
#             value_name="load"
#         )

#         self.df = df

#         print(f"[LONG] rows={len(df):,}")
#         print(f"[LONG] clients={df['client_id'].nunique()}")
#         print(f"[LONG] load max={df['load'].max():.2f}")

#         return df

#     # ======================
#     # 3. TRIM
#     # ======================
#     def trim_clients(self):
#         print("\n>>> [3/6] Trimming zeros...")

#         def trim(group):
#             group = group.sort_values("time")
#             mask = group["load"].ne(0)

#             if mask.any():
#                 first = mask.idxmax()
#                 group = group.loc[first:]

#             return group

#         self.df = self.df.groupby("client_id", group_keys=False).apply(trim)

#         print(f"[TRIM] rows={len(self.df):,}")
#         return self.df

#     # ======================
#     # 4. RESAMPLE
#     # ======================
#     def resample(self):
#         print("\n>>> [4/6] Resampling per client...")

#         df = self.df.copy()
#         df["time"] = pd.to_datetime(df["time"])
#         df = df.set_index(["client_id", "time"])

#         df = (
#             df.groupby(level=0)
#               .resample("1h", level=1)
#               .mean()
#               .reset_index()
#         )

#         df["is_zero"] = (df["load"] == 0).astype(int)

#         self.df = df

#         print(f"[RESAMPLE] rows={len(df):,}")
#         print(f"[RESAMPLE] max={df['load'].max():.2f}")
#         print(f"[RESAMPLE] p99={df['load'].quantile(0.99):.2f}")

#         return df

#     # ======================
#     # 5. FEATURES (FIXED CORE)
#     # ======================
#     def build_features(self):
#         print("\n>>> [5/6] Building features (ROBUST SCALING)...")

#         df = self.df.copy()
#         df = df.sort_values(["client_id", "time"])

#         # =========================
#         # CLIP OUTLIER FIRST
#         # =========================
#         print("[FEATURES] Clipping extreme values (0.5% tail)...")

#         def clip(x):
#             low = x.quantile(0.01)
#             high = x.quantile(0.995)
#             return x.clip(low, high)

#         df["load"] = df.groupby("client_id")["load"].transform(clip)

#         print(f"[FEATURES] After clip max={df['load'].max():.2f}")

#         # =========================
#         # ROBUST NORMALIZATION
#         # =========================
#         print("[FEATURES] Robust scaling per client...")

#         def robust_scale(x):
#             median = x.median()
#             iqr = x.quantile(0.75) - x.quantile(0.25)
#             return (x - median) / (iqr + 1e-6)

#         df["load_scaled"] = df.groupby("client_id")["load"].transform(robust_scale)

#         base = "load_scaled"

#         # =========================
#         # LAG FEATURES
#         # =========================
#         for lag in [1, 24, 48, 168]:
#             df[f"lag_{lag}"] = df.groupby("client_id")[base].shift(lag)

#         # rolling
#         df["rolling_24"] = (
#             df.groupby("client_id")[base]
#               .transform(lambda x: x.shift(1).rolling(24).mean())
#         )

#         # time features
#         df["hour"] = df["time"].dt.hour
#         df["day_of_week"] = df["time"].dt.dayofweek
#         df["month"] = df["time"].dt.month

#         df["client_code"] = df["client_id"].astype("category").cat.codes

#         # target
#         if self.use_log:
#             df["target"] = np.log1p(df["load"])
#         else:
#             df["target"] = df["load"]

#         before = len(df)
#         df = df.dropna()

#         print(f"[FEATURES] before={before:,}")
#         print(f"[FEATURES] after={len(df):,}")

#         self.df = df

#         return df

#     # ======================
#     # 6. SPLIT
#     # ======================
#     def split(self):
#         print("\n>>> [6/6] Split...")

#         df = self.df.copy()

#         df["rank"] = df.groupby("client_id")["time"].rank(pct=True)

#         train = df[df["rank"] <= 0.8]
#         test = df[df["rank"] > 0.8]

#         print(f"[SPLIT] train={len(train):,} test={len(test):,}")

#         X_train = train[self.features]
#         y_train = train["target"]

#         X_test = test[self.features]
#         y_test = test["target"]

#         return X_train, X_test, y_train, y_test

#     # ======================
#     # TRAIN
#     # ======================
#     def train(self):
#         X_train, X_test, y_train, y_test = self.split()

#         print("\n>>> Training LightGBM...")

#         self.model.fit(X_train, y_train)

#         self.X_test = X_test
#         self.y_test = y_test
#         self.y_pred = self.model.predict(X_test)

#         print(f"[TRAIN] pred min={self.y_pred.min():.4f}")
#         print(f"[TRAIN] pred max={self.y_pred.max():.4f}")

#         return self

#     # ======================
#     # EVAL
#     # ======================
#     def evaluate(self):
#         print("\n>>> Evaluating...")

#         y_true = self.y_test.copy()
#         y_pred = self.y_pred.copy()

#         if self.use_log:
#             y_true = np.expm1(y_true)
#             y_pred = np.expm1(y_pred)

#         mae = mean_absolute_error(y_true, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))

#         print("\n[RESULT]")
#         print(f"MAE : {mae:.4f}")
#         print(f"RMSE: {rmse:.4f}")

#         print("\n[EXTRA DEBUG]")
#         print(f"p95 error: {np.percentile(np.abs(y_true - y_pred), 95):.2f}")
#         print(f"p99 error: {np.percentile(np.abs(y_true - y_pred), 99):.2f}")

#         return {"MAE": mae, "RMSE": rmse}

#     # ======================
#     # RUN
#     # ======================
#     def run(self):
#         print("\n===== GLOBAL ELECTRICITY MODEL (STABLE V2) =====")

#         self.load_data()
#         self.to_long()
#         self.trim_clients()
#         self.resample()
#         self.build_features()
#         self.train()

#         return self.evaluate()

# pipe = GlobalElectricityModelDebugFixedV2(
#     dataset_path="backend/dataset/LD2011_2014.txt"
# )

# metrics = pipe.run()

# print(metrics)

# # import pandas as pd
# # import numpy as np


# # class ElectricityScaleDebugger:
# #     """
# #     DEBUG SCALE BEFORE TRAINING GLOBAL MODEL

# #     Mục tiêu:
# #     - xem scale từng client
# #     - phát hiện client lệch scale
# #     - check spike/outlier
# #     - check zero-heavy clients
# #     """

# #     def __init__(self, dataset_path: str):
# #         self.dataset_path = dataset_path
# #         self.df = None

# #     # ======================
# #     # LOAD RAW DATA
# #     # ======================
# #     def load(self):
# #         print("\n=== [LOAD DATA] ===")

# #         df = pd.read_csv(
# #             self.dataset_path,
# #             sep=";",
# #             decimal=",",
# #             parse_dates=[0]
# #         )

# #         df = df.rename(columns={"Unnamed: 0": "time"})
# #         df = df.set_index("time")

# #         self.df = df

# #         print(f"Shape: {df.shape}")
# #         print(f"Time range: {df.index.min()} → {df.index.max()}")
# #         print(f"Clients: {df.shape[1]}")
# #         print(f"Total values: {df.size:,}")

# #         return df

# #     # ======================
# #     # GLOBAL STATISTICS
# #     # ======================
# #     def global_stats(self):
# #         print("\n=== [GLOBAL STATS] ===")

# #         df = self.df

# #         print("Min load:", df.min().min())
# #         print("Max load:", df.max().max())
# #         print("Mean load:", df.mean().mean())
# #         print("Median load:", df.median().median())

# #         print("\nQuantiles (global):")
# #         flat = df.values.flatten()

# #         print("p50:", np.percentile(flat, 50))
# #         print("p90:", np.percentile(flat, 90))
# #         print("p99:", np.percentile(flat, 99))
# #         print("p99.9:", np.percentile(flat, 99.9))

# #     # ======================
# #     # ZERO ANALYSIS
# #     # ======================
# #     def zero_analysis(self):
# #         print("\n=== [ZERO ANALYSIS] ===")

# #         df = self.df

# #         zero_ratio = (df == 0).sum() / len(df)

# #         print("\nTop 10 clients nhiều zero nhất:")
# #         print(zero_ratio.sort_values(ascending=False).head(10))

# #         print("\nTop 10 clients ít zero nhất:")
# #         print(zero_ratio.sort_values().head(10))

# #     # ======================
# #     # CLIENT SCALE ANALYSIS
# #     # ======================
# #     def client_scale(self):
# #         print("\n=== [CLIENT SCALE ANALYSIS] ===")

# #         df = self.df

# #         stats = pd.DataFrame({
# #             "mean": df.mean(),
# #             "median": df.median(),
# #             "max": df.max(),
# #             "std": df.std(),
# #             "zero_ratio": (df == 0).sum() / len(df)
# #         })

# #         stats["cv"] = stats["std"] / (stats["mean"] + 1e-6)

# #         print("\nTop 10 client scale lớn nhất (mean):")
# #         print(stats.sort_values("mean", ascending=False).head(10))

# #         print("\nTop 10 client scale nhỏ nhất:")
# #         print(stats.sort_values("mean").head(10))

# #         print("\nTop 10 client biến động mạnh nhất (CV):")
# #         print(stats.sort_values("cv", ascending=False).head(10))

# #         self.stats = stats

# #         return stats

# #     # ======================
# #     # SPIKE DETECTION
# #     # ======================
# #     def spike_detection(self):
# #         print("\n=== [SPIKE ANALYSIS] ===")

# #         df = self.df

# #         flat = df.values.flatten()

# #         threshold = np.percentile(flat, 99.5)
# #         extreme = (df > threshold).sum().sum()

# #         print(f"Spike threshold (99.5%): {threshold:.2f}")
# #         print(f"Total extreme values: {extreme:,}")

# #         print("\nTop 5 clients có spike nhiều nhất:")

# #         spike_per_client = (df > threshold).sum().sort_values(ascending=False)
# #         print(spike_per_client.head(5))

# #     # ======================
# #     # SUMMARY INSIGHT
# #     # ======================
# #     def summary(self):
# #         print("\n=== [INSIGHT SUMMARY] ===")

# #         stats = self.stats

# #         print("✔ Scale imbalance check:")
# #         print(f"- max mean client: {stats['mean'].max():.2f}")
# #         print(f"- min mean client: {stats['mean'].min():.6f}")

# #         ratio = stats['mean'].max() / (stats['mean'].min() + 1e-6)

# #         print(f"\n👉 SCALE RATIO (max/min mean): {ratio:,.0f}x")

# #         if ratio > 1000:
# #             print("❌ DATA EXTREMELY IMBALANCED → MUST NORMALIZE PER CLIENT")
# #         elif ratio > 100:
# #             print("⚠️ HIGH IMBALANCE → STRONG NORMALIZATION REQUIRED")
# #         else:
# #             print("✔ SCALE OK")

# #     # ======================
# #     # RUN ALL
# #     # ======================
# #     def run(self):
# #         self.load()
# #         self.global_stats()
# #         self.zero_analysis()
# #         self.client_scale()
# #         self.spike_detection()
# #         self.summary()

# # debug = ElectricityScaleDebugger(
# #     dataset_path="backend/dataset/LD2011_2014.txt"
# # )

# # debug.run()

from backend.app.utils.preprocessor import DataChecker, Preprocessor, build_hourly_dataset
from backend.app.models.lightgbm import LightGBMTrainer

# 1. check data
# data_checker = DataChecker()
# data_checker.run_all()

# 2. tối ưu dataset
# build_hourly_dataset()

# 3. preprcess data
# preprocessor = Preprocessor(dataset_path="backend/dataset/hourly_electricity_filtered.csv")
# preprocessor.run_pipeline()

# 4. trainer
trainer = LightGBMTrainer()

model = trainer.train_model()