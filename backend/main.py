# ==== main pipeline (predict from user file) ====

from app.models.lightgbm import LightGBMTrainer
from app.models.model_manager import ModelManager

import pandas as pd

# 1. train model
trainer = LightGBMTrainer()
model = trainer.train_model()

# 2. save model
manager = ModelManager()
manager.save_model(model)

# 3. load model
loaded_model = manager.load_model()

# ==== 4. load user data ====

user_df = pd.read_csv(
    "backend/dataset/user_data.csv",
    parse_dates=["time"]
)

user_df = user_df.sort_values("time")

print("\n=== USER DATA ===")
print(user_df.head())

# ==== 5. build history ====

# lấy toàn bộ value làm history
history = user_df["value"].tolist()

# check đủ lag_7d chưa
if len(history) < 168:
    raise ValueError("User data must have at least 168 rows (7 days)")

# thời điểm bắt đầu dự đoán (sau điểm cuối cùng)
start_time = user_df["time"].iloc[-1] + pd.Timedelta(hours=1)

# ==== 6. forecast tương lai ====

future_steps = 24  # dự đoán 24 giờ tới

forecast_df = trainer.forecast_future(
    model=loaded_model,
    history=history,
    start_time=start_time,
    steps=future_steps
)

# ==== 7. combine timeline ====

# dữ liệu quá khứ
history_df = user_df.copy()
history_df = history_df.rename(columns={"value": "actual"})
history_df["prediction"] = None

# dữ liệu dự đoán
forecast_df["actual"] = None

# merge lại
timeline_df = pd.concat(
    [history_df, forecast_df],
    ignore_index=True
)

timeline_df = timeline_df.sort_values("time")

# ==== 8. output ====

print("\n=== FULL TIMELINE ===\n")
print(timeline_df.tail(30))