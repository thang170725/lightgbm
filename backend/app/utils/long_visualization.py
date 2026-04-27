import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.plotting import autocorrelation_plot

class ElectricityDataService:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:

            current_dir = os.path.dirname(os.path.abspath(__file__))

            backend_dir = os.path.dirname(os.path.dirname(current_dir))
            dataset_path = os.path.join(backend_dir, "dataset", "LD2011_2014.txt")
            
        self.df = pd.read_csv(
            dataset_path,
            sep=";",
            decimal=",",
            parse_dates=[0]
        )
        self.df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        
    def get_random_users_data(self, size_rnd=9):
        """Pick random customers (users) and return their data."""
        all_users = [col for col in self.df.columns if col != "timestamp"]
        random_users = np.random.choice(all_users, size=size_rnd, replace=False)
        
        cols_to_keep = ["timestamp"] + list(random_users)
        return self.df[cols_to_keep], random_users

class BoxplotService(ElectricityDataService):
    """Service dedicated to drawing Boxplots."""
    def plot_random_users(
        self,
        size_rnd=9,
        figsize=(14, 8),
        save=False,
        save_path="./backend/dataset/images",
        show=True
    ):
        
        df_users, random_users = self.get_random_users_data(size_rnd)
        
        
        df_melted = pd.melt(
            df_users, 
            id_vars=["timestamp"], 
            value_vars=random_users, 
            var_name="customer", 
            value_name="load"
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df_melted, x="customer", y="load", ax=ax, palette="Set3", hue="customer", legend=False)
        
        ax.set_xlabel("", fontsize=12)
        ax.set_ylabel("Điện năng tiêu thụ", fontsize=12)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/users_boxplot_random.png",
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()
        plt.close(fig)

class ACFService(ElectricityDataService):
    """Service dedicated to drawing ACF plots."""
    def plot_random_users(
        self,
        size_rnd=9,                  
        figsize=(16, 12),
        rc_chart=(3, 3),
        save=False,
        save_path="./backend/dataset/images",
        show=True,
        resample_daily=True
    ):
        df_users, random_users = self.get_random_users_data(size_rnd)

        if resample_daily:
            df_users.set_index("timestamp", inplace=True)
            df_plot = df_users.resample("D").sum()
        else:
            df_plot = df_users.set_index("timestamp")
        
        fig, axes = plt.subplots(rc_chart[0], rc_chart[1], figsize=figsize)
        axes = axes.flatten()
        
        for i, user in enumerate(random_users):
            if i < len(axes):
                autocorrelation_plot(df_plot[user], ax=axes[i])
                axes[i].set_title(f"Khách hàng: {user}")
                axes[i].set_xlabel("Độ trễ (Lag)" if resample_daily else "Độ trễ (Lag)")
                axes[i].set_ylabel("Hệ số ACF")
            
        for j in range(len(random_users), len(axes)):
            axes[j].axis("off")
            
        plt.suptitle(f"Biểu đồ ACF", fontsize=18, y=1.02)
        plt.tight_layout()
        
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/users_acf_random.png",
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()
        plt.close(fig)

class MonthlyAggregateService(ElectricityDataService):

    def plot_9_users_individual_boxplots(self, size_rnd=9, figsize=(16, 12), rc_chart=(3, 3), save=False, save_path="./backend/dataset/images", show=True):
        # Chọn 9 khách hàng ngẫu nhiên
        users = [col for col in self.df.columns if col != "timestamp"]
        random_users = np.random.choice(users, size=size_rnd, replace=False)
        
        # Lọc dữ liệu từ 1/1/2011 đến 1/1/2015
        start_date = pd.to_datetime('2011-01-01')
        end_date = pd.to_datetime('2015-01-01')
        mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
        df_filtered = self.df[mask].copy()
        
        # Lấy riêng cột timestamp và 9 khách hàng
        df_selected = df_filtered[['timestamp'] + list(random_users)].copy()
        
        # Gộp dữ liệu theo ngày để boxplot gọn hơn
        df_selected.set_index("timestamp", inplace=True)
        df_daily = df_selected.resample('D').sum().reset_index()
        
        # Thêm cột Tháng (1-12)
        df_daily['month'] = df_daily['timestamp'].dt.month
        
        # Vẽ 9 boxplot riêng biệt
        fig, axes = plt.subplots(rc_chart[0], rc_chart[1], figsize=figsize)
        axes = axes.flatten()
        
        for i, user in enumerate(random_users):
            sns.boxplot(data=df_daily, x='month', y=user, ax=axes[i], palette='Set3', hue='month', legend=False)
            axes[i].set_title(user, fontsize=12)
            axes[i].set_xlabel("Tháng")
            axes[i].set_ylabel("Tổng tiêu thụ")
            
        plt.suptitle("Biểu đồ Boxplot", fontsize=16)
        plt.tight_layout()
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/monthly_9users_individual_boxplots.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


    def plot_9_users_individual_acf(self, size_rnd=9, figsize=(16, 12), rc_chart=(3, 3), save=False, save_path="./backend/dataset/images", show=True):
        # Chọn 9 mt
        users = [col for col in self.df.columns if col != "timestamp"]
        random_users = np.random.choice(users, size=size_rnd, replace=False)
        
        # Lọc dữ liệu từ 1/1/2011 đến 1/1/2015
        start_date = pd.to_datetime('2011-01-01')
        end_date = pd.to_datetime('2015-01-01')
        mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
        df_filtered = self.df[mask].copy()
        
        # Lấy riêng cột timestamp và 9 khách hàng
        df_selected = df_filtered[['timestamp'] + list(random_users)].copy()
        
        # Gộp dữ liệu thành tổng tải theo từng tháng cho mỗi khách hàng
        df_selected.set_index("timestamp", inplace=True)
        df_monthly = df_selected.resample('ME').sum()
        
        # Vẽ 9 ACF 
        fig, axes = plt.subplots(rc_chart[0], rc_chart[1], figsize=figsize)
        axes = axes.flatten()
        
        for i, user in enumerate(random_users):
            autocorrelation_plot(df_monthly[user], ax=axes[i])
            axes[i].set_title(user, fontsize=12)
            axes[i].set_xlabel("Độ trễ ")
            axes[i].set_ylabel("Hệ số ACF")
            
        plt.suptitle("Biểu đồ ACF", fontsize=16)
        plt.tight_layout()
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/monthly_9users_individual_acf.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    print("Drawing Boxplots...")
    monthly_service = MonthlyAggregateService()
    monthly_service.plot_9_users_individual_boxplots(save=True, show=True)
    print("Saved")
    
    print("Drawing ACFs...")
    monthly_service.plot_9_users_individual_acf(save=True, show=True)
    print("Saved")
