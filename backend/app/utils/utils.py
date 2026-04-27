import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualization:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(current_dir))
            dataset_path = os.path.join(backend_dir, "LD2011_2014.txt")
            
        self.df = pd.read_csv(
            dataset_path,
            sep=";",
            decimal=",",
            parse_dates=[0]
        )

        self.columns = self.df.columns
        self.rows = len(self.df)

    def plot_chart(self, 
        step:int,                                       
        images: int,                                    
        size_rnd: int,                              # số lượng biểu đồ muốn tạo (18: muốn tạo 18 cái biểu đồ)                       
        figsize: tuple,                                 
        rc_chart: tuple,                            # số hàng cột trong 1 images ((3,3): tạo ra 1 ảnh bên trong có 9 buổi đồ 3x3)        
        save:bool = False,                              
        save_path: str = "backend/dataset/images"     
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
                    x=sampled["Unnamed: 0"], 
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
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/users_page_{page+1}.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)
    
    def plot_aggregate_load_chart(
        self,
        size_rnd=9,                  
        figsize=(14,8),
        rc_chart=(3,3),
        save=False,
        save_path="./backend/dataset/images"        
    ):
        df = self.df.copy()

        df["Unnamed: 0"] = pd.to_datetime(
            df["Unnamed: 0"]
        )

        df["aggregate"] = df.iloc[:,1:].sum(axis=1)

        df["date"] = df["Unnamed: 0"].dt.date
        df["hour"] = df["Unnamed: 0"].dt.hour + \
                     df["Unnamed: 0"].dt.minute/60

        all_days = df["date"].unique()

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

        for j in range(len(random_days), len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Aggregate Load Profiles of Random Days",
            fontsize=14
        )

        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/aggregate_random_days.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

    def plot_time_pattern_heatmap(
        self,
        figsize=(15, 8),
        save=False,
        save_path="./backend/dataset/images"
    ):
        df: pd.DataFrame = self.df.copy()

        df["Unnamed: 0"] = pd.to_datetime(
            df["Unnamed: 0"]
        )

        df["aggregate"] = df.iloc[:,1:].sum(axis=1) # tổng điện năng của 370 người mỗi một khung giờ

        df["hour"] = df["Unnamed: 0"].dt.hour
        df["day_name"] = df["Unnamed: 0"].dt.day_name()
        df["day_of_week"] = df["Unnamed: 0"].dt.dayofweek

        pivot_df = df.groupby(['day_name', 'day_of_week', 'hour'])['aggregate'].mean().reset_index()

        heatmap_data = pivot_df.pivot(index='day_name', columns='hour', values='aggregate')

        diff = heatmap_data.mean()
        heatmap_data_diff = heatmap_data - diff

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_order = [day for day in days_order if day in heatmap_data.index]
        heatmap_data_diff = heatmap_data_diff.reindex(days_order)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            heatmap_data_diff, 
            cmap='RdBu_r', 
            annot=False, 
            linewidths=.5, 
            cbar_kws={'label': 'Difference from Average Load'},
            ax=ax
        )

        ax.set_title('Time-Pattern Heatmap of Aggregate Load', fontsize=16, pad=20)
        ax.set_xlabel('Hour of Day (0-23)', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        
        plt.yticks(rotation=0)
        
        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/time_pattern_heatmap.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

    def plot_histogram(
        self,
        figsize=(12, 6),
        save=False,
        save_path="./backend/dataset/images"
    ):
        df = self.df.copy()

        df["aggregate"] = df.iloc[:,1:].sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(
            data=df, 
            x="aggregate", 
            bins=50, 
            kde=True, 
            color='skyblue',
            ax=ax
        )
        
        mean_val = df["aggregate"].mean()
        median_val = df["aggregate"].median()
        
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_title('Distribution of Aggregate Load', fontsize=16, pad=20)
        ax.set_xlabel('Aggregate Load', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/aggregate_histogram.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    visual = Visualization(dataset_path="./backend/dataset/LD2011_2014.txt")
    
    # 1. Gọi thử Time-pattern Heatmap
    visual.plot_time_pattern_heatmap(
        figsize=(15, 8),
        save=True
    )

    # 2. Gọi thử Histogram
    visual.plot_histogram(
        figsize=(12, 6),
        save=True
    )
