import os
import joblib

class ModelManager:
    def __init__(self,
        model_dir="backend/models"
    ):
        self.model_dir = model_dir

        # tạo folder nếu chưa có
        os.makedirs(self.model_dir, exist_ok=True)

    # ====== save model =======
    def save_model(self,
        model,
        evaluate: dict,
        save_path="lightgbm_model.pkl"
    ):
        path = os.path.join(self.model_dir, save_path)

        joblib.dump({
            'model': model,
            'evaluate': evaluate
        }, path)

        print(f"\nModel saved to: {path}")

    # load model
    def load_model(self,
        name="lightgbm_model.pkl"
    ):
        path = os.path.join(self.model_dir, name)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        print(f"\nLoading model from: {path}")

        return joblib.load(path)