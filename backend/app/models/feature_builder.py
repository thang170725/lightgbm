import numpy as np
import pandas as pd

class FeatureBuilder:
    def __init__(self):
        pass

    # ==== time features ====
    def build_time_features(
        self,
        time_str
    ):
        dt = pd.to_datetime(time_str)

        hour = dt.hour

        day_of_week = dt.dayofweek

        is_weekend = (
            1 if day_of_week >= 5 else 0
        )

        # cyclical encoding
        hour_sin = np.sin(
            2 * np.pi * hour / 24
        )

        hour_cos = np.cos(
            2 * np.pi * hour / 24
        )

        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }

    # ==== lag features ====
    def build_lag_features(
        self,
        history
    ):
        """
        history:
            list giá trị theo thời gian
            (đã sort tăng dần)
        """

        if len(history) < 168:
            raise ValueError(
                "History phải >= 168 điểm (7 ngày)"
            )

        # ===== lag =====
        lag_24h = history[-24]

        lag_48h = history[-48]

        lag_72h = history[-72]

        lag_7d = history[-168]

        # ===== rolling =====
        rolling_mean_24h = np.mean(
            history[-24:]
        )

        rolling_mean_7d = np.mean(
            history[-168:]
        )

        rolling_std_24h = np.std(
            history[-24:]
        )

        return {
            # lag
            "lag_24h": np.log1p(lag_24h),
            "lag_48h": np.log1p(lag_48h),
            "lag_72h": np.log1p(lag_72h),
            "lag_7d": np.log1p(lag_7d),

            # rolling
            "rolling_mean_24h": np.log1p(
                rolling_mean_24h
            ),

            "rolling_mean_7d": np.log1p(
                rolling_mean_7d
            ),

            "rolling_std_24h": np.log1p(
                rolling_std_24h
            )
        }

    # ==== build all features ====
    def build_features(
        self,
        time_str,
        history
    ):
        features = {}

        # time features
        features.update(
            self.build_time_features(
                time_str
            )
        )

        # lag + rolling features
        features.update(
            self.build_lag_features(
                history
            )
        )

        return features