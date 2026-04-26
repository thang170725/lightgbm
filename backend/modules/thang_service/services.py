import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    def plot_chart(self, 
        step:int,                                       # bước nhảy để vẽ biểu đồ cho đẹp và tối ưu tốc độ
        images: int,                                    # chỉ định vẽ làm mấy ảnh 
        size_rnd: int,                                  # trong số ảnh đó vẽ tổng mấy biểu đồ
        figsize: tuple,                                 # kích thước của 1 biểu đồ
        rc_chart: tuple,                                # số hàng cột trong 1 image
        save:bool = False,                              # có lưu ảnh hay không
        save_path: str = "./backend/dataset/images"     # chỉ định lưu ở đâu
    ):
        users = np.random.choice(self.columns[1:], size=size_rnd, replace=False)                

        sampled = self.df.iloc[::step]

        per_page = rc_chart[0]*rc_chart[1]
        for page in range(images):
            subset = users[page*per_page:(page+1)*per_page]

            fig, axes = plt.subplots(
                rc_chart[0],rc_chart[1],
                figsize=figsize
            )
            axes = axes.flatten()

            for i, user in enumerate(subset):
                sns.lineplot(
                    x=sampled["Unnamed: 0"], # time,
                    y=sampled[user],
                    ax=axes[i]
                )
                x = sampled["Unnamed: 0"]

                axes[i].set_title(user)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
                axes[i].set_xticks([x.iloc[0], x.iloc[-1]])

            plt.tight_layout()
            if save:
                plt.savefig(f"{save_path}/users_page_{page+1}.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)
    
    def plot_aggregate_load_chart(
        self,
        size_rnd=9,                  # số ngày random
        figsize=(14,8),
        rc_chart=(3,3),
        save=False,
        save_path="./backend/dataset/images"        
    ):
        df = self.df.copy()

        df["Unnamed: 0"] = pd.to_datetime(
            df["Unnamed: 0"]
        )

        # tổng 370 users
        df["aggregate"] = df.iloc[:,1:].sum(axis=1)

        # tách ngày
        df["date"] = df["Unnamed: 0"].dt.date
        df["hour"] = df["Unnamed: 0"].dt.hour + \
                     df["Unnamed: 0"].dt.minute/60

        # danh sách ngày unique
        all_days = df["date"].unique()

        # random ngày
        random_days = np.random.choice(
            all_days,
            size=size_rnd,
            replace=False
        )

        fig, axes = plt.subplots(
            rc_chart[0],
            rc_chart[1],
            figsize=figsize
        )

        axes = axes.flatten()

        for i, day in enumerate(random_days):

            day_data = df[
                df["date"] == day
            ]

            sns.lineplot(
                x=day_data["hour"],
                y=day_data["aggregate"],
                ax=axes[i]
            )

            axes[i].set_title(str(day))
            axes[i].set_xlim(0,24)

            axes[i].set_xticks(
                [0,6,12,18,24]
            )

            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

        # tắt subplot thừa
        for j in range(len(random_days), len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Aggregate Load Profiles of Random Days",
            fontsize=14
        )

        plt.tight_layout()

        if save:
            plt.savefig(
                f"{save_path}/aggregate_random_days.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        
if __name__ == "__main__":
    visual = Visualization()
    # visual.plot_chart(
    #     step=750,
    #     images=2,
    #     size_rnd=18,
    #     figsize=(12,5),
    #     rc_chart=(3,3),
    #     save=True
    # )
    visual.plot_aggregate_load_chart(
        size_rnd=9,
        rc_chart=(3,3),
        save=True
    )
