import pandas as pd

class Preprocessor:
    def __init__(self, dataset_path:str="backend/dataset/LD2011_2014.txt"):
        self.df = pd.read_csv(
            dataset_path,
            sep=';',
            decimal=',',
            parse_dates=[0]
        )
    
    # kiểm tra khách hàng nào có nhiều số 0 bất thường
    def check_zero_user(self):
        