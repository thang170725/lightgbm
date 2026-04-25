import pandas as pd

class Visualization:
    def __init__(self, dataset_path: str = "./backend/dataset/LD2011_2014.txt"):
        self.df = pd.read_csv(
            dataset_path,
            sep=";",
            decimal=",",
            parse_dates=[0]
        )

        self.columns = self.df.columns
        self.rows = len(self.df)
        print(self.columns, self.rows)

if __name__ == "__main__":
    visual = Visualization()
