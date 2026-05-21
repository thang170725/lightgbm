from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from backend.app.models.lightgbm import LightGBMTrainer
from backend.app.models.model_manager import ModelManager

app = FastAPI()

# ==== 1. create =====
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
loader = manager.load_model(name="lightgbm_model_v2.pkl")
model = loader['model']
evaluate = loader['evaluate']

# ==== API post ====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # đọc csv upload
        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")), 
            parse_dates=["time"]
        )
        df = df.sort_values("time")

        # lấy history
        history = df["value"].tolist()

        if len(history) < 168:
            return {"error": "Need at least 168 rows"}

        start_time = df["time"].iloc[-1] + pd.Timedelta(hours=1) # để tạo timestamp mới nhất và cộng thêm 1 giờ để bắt đầu dự đoán

        forecast_df = trainer.forecast_future(
            model=model,
            history=history,
            start_time=start_time,
            steps=24
        )

        # === convert data ===
        forecast_df = forecast_df.rename(columns={"prediction": "value"})

        result = forecast_df[["time", "value"]].copy()
        result = result.where(pd.notnull(result), None)

        return {
            'result': result.to_dict(orient="records"),
            'evaluate': evaluate
        }
    except Exception as e:
        return {
            "error": str(e)
        }