import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from backend.app.models.feature_builder import FeatureBuilder

# ==== model ai/ml lightgbm ====
class LightGBMTrainer:
    def __init__(self,
        train_path="backend/dataset/train_ready.csv",
        valid_path="backend/dataset/valid_ready.csv",
        test_path="backend/dataset/test_ready.csv"
    ):
        print("Loading datasets (train, valid, test)...")

        self.train = pd.read_csv(train_path)
        self.valid = pd.read_csv(valid_path)
        self.test = pd.read_csv(test_path)

        self.features = [
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "lag_24h",
            "lag_7d",
            "rolling_mean_24h"
        ]
        self.target = "target_value"

    # data
    def get_data(self):
        print("\nPreparing data...")

        steps = ["Train", "Valid", "Test"]
        data = []

        # tqdm cho quá trình split
        for step in tqdm(steps, desc="Splitting data"):
            if step == "Train":
                data.append((self.train[self.features], self.train[self.target]))
            elif step == "Valid":
                data.append((self.valid[self.features], self.valid[self.target]))
            else:
                data.append((self.test[self.features], self.test[self.target]))

        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data

        return (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        )

    # model
    def build_model(self):
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        return model

    # metrics
    def evaluate(self,
        y_true,
        preds
    ):
        # inverse log1p
        y_true = np.expm1(y_true)
        preds = np.expm1(preds)

        mae = mean_absolute_error(y_true, preds)
        relative_mae = (
        mae / np.mean(y_true)
        ) * 100

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                preds
            )
        )
        mape = np.mean(
            np.abs(
                (
                    y_true-preds
                )/y_true
            )
        ) * 100
        r2 = r2_score(y_true, preds)

        print(f"MAE: {mae:.4f}")
        print(f"Relative MAE: {relative_mae:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R2 SCORE: {r2:.2f}%")

        return {
            "mae": mae,
            "relative_mae": relative_mae,
            "mape": mape,
            "rmse": rmse,
            "r2": r2
        }

    # train
    def train_model(self):
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        ) = self.get_data()

        model = self.build_model()

        print("\nTraining model...")

        # tqdm wrapper (LightGBM không expose progress trực tiếp)
        for _ in tqdm(range(1), desc="Fitting LightGBM"):
            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (
                        X_valid,
                        y_valid
                    )
                ],
                eval_metric="l2",
                callbacks=[
                    lgb.log_evaluation(period=50),  # in log mỗi 50 dòng,
                    lgb.early_stopping(stopping_rounds=30)
                ]
            )

        print(f"Best iteration: {model.best_iteration_}")

        print("\nPredicting...")
        preds = model.predict(X_test)

        print("\n=== Test Metrics ===")
        evaluate = self.evaluate(y_test, preds)

        importance = pd.DataFrame({
            "feature": self.features,
            "importance": model.feature_importances_
        }).sort_values(
            "importance",
            ascending=False
        )

        print("\nFeature importance:\n", importance)

        return model, evaluate

    # ==== predict new data ====
    def preprocess_input(self, input_dict):
        """
        input_dict: dữ liệu user nhập dạng dict
        ví dụ:
        {
            "hour_sin": 0.5,
            "hour_cos": 0.86,
            "day_of_week": 2,
            "lag_24h": 120,
            "lag_7d": 130,
            "rolling_mean_24h": 125
        }
        """

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[self.features]

        return df


    def predict_new(self, model, input_dict):
        """
        predict cho 1 sample mới
        """

        X_new = self.preprocess_input(input_dict)

        preds = model.predict(X_new)

        # inverse log1p (giống evaluate)
        preds = np.expm1(preds)

        return preds[0]
    
    def forecast_future(self, model, history, start_time, steps):
        """
        history: list giá trị quá khứ (>=168)
        start_time: thời điểm bắt đầu dự đoán
        steps: số bước tương lai (vd: 24 giờ)
        """

        results = []
        history = history.copy()

        current_time = pd.to_datetime(start_time)

        for i in range(steps):
            # build feature từ history hiện tại
            features = FeatureBuilder().build_features(
                time_str=current_time,
                history=history
            )

            pred = self.predict_new(model, features)

            results.append({
                "time": current_time,
                "prediction": float(pred)
            })

            # 🔥 cực quan trọng: append prediction vào history
            history.append(pred)

            # tăng thời gian (1 giờ)
            current_time += pd.Timedelta(hours=1)

        return pd.DataFrame(results)