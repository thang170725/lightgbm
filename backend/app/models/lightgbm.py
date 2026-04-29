import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


class LightGBMTrainer:

    def __init__(
        self,
        train_path="backend/dataset/train_ready.csv",
        valid_path="backend/dataset/valid_ready.csv",
        test_path="backend/dataset/test_ready.csv"
    ):

        self.train = pd.read_csv(
            train_path
        )

        self.valid = pd.read_csv(
            valid_path
        )

        self.test = pd.read_csv(
            test_path
        )


        self.features = [
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "lag_24h",
            "lag_7d",
            "rolling_mean_24h"
        ]

        self.target = "target_value"



    # -------------------------
    # data
    # -------------------------
    def get_data(self):

        X_train = self.train[
            self.features
        ]

        y_train = self.train[
            self.target
        ]


        X_valid = self.valid[
            self.features
        ]

        y_valid = self.valid[
            self.target
        ]


        X_test = self.test[
            self.features
        ]

        y_test = self.test[
            self.target
        ]

        return (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        )



    # -------------------------
    # model
    # -------------------------
    def build_model(
        self
    ):

        model = lgb.LGBMRegressor(

            objective="regression",

            n_estimators=300,

            learning_rate=0.05,

            num_leaves=31,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42,

            n_jobs=-1
        )

        return model



    # -------------------------
    # metrics
    # -------------------------
    def evaluate(
        self,
        y_true,
        preds
    ):

        # inverse log1p
        y_true = np.expm1(
            y_true
        )

        preds = np.expm1(
            preds
        )


        mae = mean_absolute_error(
            y_true,
            preds
        )

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


        print(
            f"MAE: {mae:.4f}"
        )

        print(
            f"RMSE: {rmse:.4f}"
        )

        print(
            f"MAPE: {mape:.2f}%"
        )



    # -------------------------
    # train
    # -------------------------
    def train_model(
        self
    ):

        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test

        ) = self.get_data()


        model = self.build_model()


        model.fit(
            X_train,
            y_train,

            eval_set=[
                (
                    X_valid,
                    y_valid
                )
            ],

            eval_metric="l2"
        )


        preds = model.predict(
            X_test
        )


        print(
            "\n=== Test Metrics ==="
        )

        self.evaluate(
            y_test,
            preds
        )


        importance = pd.DataFrame({

            "feature":
            self.features,

            "importance":
            model.feature_importances_

        }).sort_values(
            "importance",
            ascending=False
        )


        print(
            "\nFeature importance:"
        )

        print(
            importance
        )


        return model