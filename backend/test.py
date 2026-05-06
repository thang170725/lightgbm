from backend.app.utils.preprocessor import DataChecker, Preprocessor, build_hourly_dataset
from backend.app.models.lightgbm import LightGBMTrainer

# 1. check data
# data_checker = DataChecker()
# data_checker.run_all()

# 2. tối ưu dataset
# build_hourly_dataset()

# 3. preprcess data
preprocessor = Preprocessor(dataset_path="backend/dataset/hourly_electricity_filtered.csv")
preprocessor.run_pipeline()

# 4. trainer
# trainer = LightGBMTrainer()

# model = trainer.train_model()