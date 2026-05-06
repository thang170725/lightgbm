import numpy as np
import pandas as pd

class FeatureBuilder:
    def __init__(self):
        pass

    def build_time_features(self, time_str):
        dt = pd.to_datetime(time_str)

        hour = dt.hour
        day_of_week = dt.dayofweek

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week
        }

    def build_lag_features(self, history):
        """
        history: list giá trị theo thời gian (đã sort tăng dần)
        """

        if len(history) < 168:
            raise ValueError("History phải >= 168 điểm (7 ngày)")

        return {
            "lag_24h": history[-24],
            "lag_7d": history[-168],
            "rolling_mean_24h": np.mean(history[-24:])
        }

    def build_features(self, time_str, history):
        features = {}

        features.update(self.build_time_features(time_str))
        features.update(self.build_lag_features(history))

        return features