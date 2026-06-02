from backend.app.utils.preprocessor import Preprocessor
from backend.app.models.lightgbm import LightGBMTrainer
from backend.app.models.model_manager import ModelManager

# 1. check data
# data_checker = DataChecker()
# data_checker.run_all()

# 3. preprcess data
# preprocessor = Preprocessor(dataset_path="backend/dataset/hourly_electricity_filtered.csv")
# preprocessor.run_pipeline()

# 4. trainer
# trainer = LightGBMTrainer()
# model, evaluate = trainer.train_model()

# 5. load model
# model_manager = ModelManager(model_dir='backend/models')
# save = model_manager.save_model(
#     model,
#     evaluate,
#     save_path='lightgbm_model.pkl'
# )
# loader = model_manager.load_model(name='lightgbm_model_v2.pkl')
# print(loader)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

t = np.arange(100)

series = pd.Series(
    np.sin(2*np.pi*t/12)
)

autocorrelation_plot(series)

plt.show()