from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from backend.app.models.lightgbm import LightGBMTrainer
from backend.app.models.model_manager import ModelManager

app = FastAPI()

# 👇 thêm đoạn này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev thì để *, prod thì phải fix domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trainer = LightGBMTrainer()
manager = ModelManager()

# load model 1 lần
# model = trainer.train_model()
# manager.save_model(model)
loader = manager.load_model()
model = loader['model']
evaluate = loader['evaluate']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    df = pd.read_csv(io.StringIO(contents.decode("utf-8")), parse_dates=["time"])
    df = df.sort_values("time")

    history = df["value"].tolist()

    if len(history) < 168:
        return {"error": "Need at least 168 rows"}

    start_time = df["time"].iloc[-1] + pd.Timedelta(hours=1)

    forecast_df = trainer.forecast_future(
        model=model,
        history=history,
        start_time=start_time,
        steps=24
    )

    # 👉 CHỈ TRẢ PREDICTION
    forecast_df = forecast_df.rename(columns={"prediction": "value"})

    result = forecast_df[["time", "value"]].copy()

    result = result.where(pd.notnull(result), None)

    return {
        'result': result.to_dict(orient="records"),
        'evaluate': evaluate
    }