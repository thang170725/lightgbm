import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class VisualizationOriginalDataset:
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

    # biểu đồ đường mô tả sự biến thiên về mức sử dụng điện của các user
    # dùng để xem mức phân bổ sử dụng điện của user
    def dayly_load_chart(self,                                     
        images: int,                                    
        size_rnd: int,                              # số lượng biểu đồ muốn tạo (18: muốn tạo 18 cái biểu đồ)                       
        space_chart: tuple,                         # khoảng cách của các biểu đồ trong cùng một page (w,h)        
        rc_chart: tuple,                            # số hàng cột trong 1 images ((3,3): tạo ra 1 ảnh bên trong có 9 buổi đồ 3x3)        
        save:bool = False,                              
        save_path: str = "backend/dataset/images"     
    ):
        users = np.random.choice(self.columns[1:], size=size_rnd, replace=False)                

        daily_df = (
            self.df
            .set_index("timestamp")
            .resample("5D")
            .mean()
            .reset_index()
        )

        per_page = rc_chart[0]*rc_chart[1]
        for page in range(images):
            subset = users[page*per_page:(page+1)*per_page]

            fig, axes = plt.subplots(
                rc_chart[0],rc_chart[1],
            )
            axes = axes.flatten()

            for i, user in enumerate(subset):
                sns.lineplot(
                    x=daily_df["timestamp"], 
                    y=daily_df[user],
                    ax=axes[i],
                    linewidth=0.5,
                    color="black"
                )
                x = daily_df["timestamp"]

                axes[i].set_title(user)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
                axes[i].set_xticks([x.iloc[0], x.iloc[-1]])
                axes[i].tick_params(
                    axis="both",
                    labelsize=5
                )
                axes[i].tick_params(
                    axis="x",
                    rotation=30
                )

            plt.subplots_adjust(
                wspace=space_chart[0],
                hspace=space_chart[1]
            )
            if save:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/users_page_{page+1}.png", dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close(fig)
    
    # biểu đồ đường xem sự biến thiên tổng lượng điện năng tiêu thụ trong 24h của 370 người
    def plot_aggregate_load_chart(
        self,
        size_rnd=9,                  
        rc_chart=(3,3),
        save=False,
        save_path="./backend/dataset/images"        
    ):
        df = self.df.copy()

        df["Unnamed: 0"] = pd.to_datetime(
            df["timestamp"]
        )

        df["aggregate"] = df.iloc[:,1:].sum(axis=1)

        df["date"] = df["Unnamed: 0"].dt.date
        df["hour"] = df["Unnamed: 0"].dt.hour + df["Unnamed: 0"].dt.minute/60 # để làm trúc X

        all_days = df["date"].unique()

        random_days = np.random.choice(
            all_days,
            size=size_rnd,
            replace=False
        )

        fig, axes = plt.subplots(
            rc_chart[0],rc_chart[1],
        )

        axes = axes.flatten()

        for i, day in enumerate(random_days):
            day_data = df[
                df["date"] == day
            ]

            sns.lineplot(
                x=day_data["hour"],
                y=day_data["aggregate"],
                ax=axes[i],
                color="black"
            )

            axes[i].set_title(str(day))
            axes[i].set_xlim(0,24)

            axes[i].set_xticks(
                [0,6,12,18,24]
            )

            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].tick_params(
                axis="both",
                labelsize=5
            )

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

        else: 
            plt.show()
        plt.close(fig)

    # biểu đồ độ tương quan
    def plot_time_pattern_heatmap(
        self,
        figsize=(15, 8),
        save=False,
        save_path="./backend/dataset/images"
    ):
        df: pd.DataFrame = self.df.copy()

        df["Unnamed: 0"] = pd.to_datetime(
            df["timestamp"]
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
            heatmap_data, 
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
                f"{save_path}/time_pattern_heatmap_original.png",
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
    visual = Visualization(dataset_path="./backend/dataset/hourly_electricity_filtered.csv")
    visual.dayly_load_chart(
        images=3,
        size_rnd=27,
        space_chart=(0.5,0.5),
        rc_chart=(3,3),
        save=False
    )

    visual.plot_aggregate_load_chart(
        size_rnd=9,
        rc_chart=(3,3),
        save=False
    )
    
    # 1. Gọi thử Time-pattern Heatmap
    visual.plot_time_pattern_heatmap(
        figsize=(15, 8),
        save=False
    )

    # # 2. Gọi thử Histogram
    visual.plot_histogram(
        figsize=(12, 6),
        save=False
    )
