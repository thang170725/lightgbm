import pandas as pd
import numpy as np

# === class check data ===
# use this class before running class Preprocessor 
class DataChecker:
    """
    Check chất lượng dataset điện năng trước khi train model.
    Không modify data, chỉ phân tích.
    """
    def __init__(self, dataset_path:str="backend/dataset/LD2011_2014.txt"):
        self.df = pd.read_csv(
            dataset_path,
            sep=';',
            decimal=',',
            parse_dates=[0]
        )
    
    # 1. Tổng quan về dataset
    def overview(self):
        print("\n=== DATA OVERVIEW ===")
        print("Shape:", self.df.shape)
        print("Rows: ", self.df.shape[0])
        print("Columns:", self.df.shape[1])
    
    # 2. Missing values
    def check_missing(self):
        print("\n=== MISSING VALUES ===")

        missing = self.df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("No missing values.")
        else:
            print(missing)
    
       # -------------------------
    
    # 3. Zero ratio per client
    def check_zero_ratio(self):
        print("\n=== ZERO RATIO PER CLIENT ===")

        zero_ratio = (self.df == 0).mean().sort_values(ascending=False)

        print("Top 10 clients with most zeros:")
        print(zero_ratio.head(10))

        print("\nMost active clients:")
        print(zero_ratio.tail(10))
    
    # 4. kiểm tra chuỗi 0 liên tục
    def max_consecutive_zeros(self):
        '''
        Cái này quan trọng hơn count đơn thuần.
        Ví dụ:
            1-2 điểm zero lẻ tẻ → có thể bình thường
            500 bước liên tục zero → bất thường.
        '''
        print("\n=== Check for a string of consecutive zeros ===")
        for col in self.df.columns[:20]:
            groups = (self.df[col] != 0).cumsum()

            print(col, self.df[col].eq(0).groupby(groups).sum().max())

    # 5. Detect "inactive period"
    def detect_activation_time(self):
        print("\n=== FIRST NON-ZERO (CLIENT ACTIVATION TIME) ===")

        activation = {}

        for col in self.df.columns:
            series = self.df[col]

            if (series != 0).sum() == 0:
                activation[col] = None
                continue

            first_active = series.ne(0).idxmax()
            activation[col] = first_active

        activation_series = pd.Series(activation).dropna()

        print("Sample activation times:")
        print(activation_series.head(10))

    # 6. Sparsity analysis
    def sparsity_report(self):
        print("\n=== SPARSITY REPORT ===")

        total_values = self.df.size
        zero_values = (self.df == 0).sum().sum()

        sparsity = zero_values / total_values

        print(f"Total values: {total_values}")
        print(f"Zero values: {zero_values}")
        print(f"Sparsity: {sparsity:.4f}")

    # 7. Time frequency check
    def check_time_frequency(self):
        print("\n=== TIME FREQUENCY CHECK ===")

        diff = self.df.index.to_series().diff().value_counts()

        print(diff.head(10))

    # RUN ALL
    def run_all(self):
        self.overview()
        self.check_missing()
        self.check_zero_ratio()
        self.max_consecutive_zeros()
        self.detect_activation_time()
        self.sparsity_report()
        self.check_time_frequency()

# === tạo dataset hourly và lưu ra csv ===
# vì dataset gốc quá lớn nên cần cắt giảm để tối ưu hơn
def build_hourly_dataset(
    input_path="backend/dataset/LD2011_2014.txt",                   # đường dẫn dataset gốc
    output_path="backend/dataset/hourly_electricity_filtered.csv",  # đường dẫn dataset sau khi xử lý (lưu ý đuôi file là .csv)
    top_clients=180,                                                # lấy 180 người tốt nhất thay vì 370 người của dataset ban đầu
    min_mean_load=1.0,
    keep_year="2014"                                                # chỉ giữ 1 năm cuối
):
    df = pd.read_csv(
        input_path,
        sep=";",
        decimal=",",
        parse_dates=[0]
    )

    df.rename(  # đổi tên cột
        columns={
            "Unnamed: 0": "timestamp"
        },
        inplace=True
    )

    df.set_index(
        "timestamp",
        inplace=True
    )

    # memory optimize
    df = df.astype(
        "float32"
    )

    df = df / 4
    
    # 15 phút -> 1 giờ
    df_hourly = (
        df
        .resample("1h")
        .sum()
    )

    # chỉ giữ 1 năm
    df_hourly = df_hourly.loc[
        keep_year
    ]

    # in ra để debug
    print("after year filter:", df_hourly.shape)

    # ==== chọn active clients ====
    # tính phân trăm người dùng có mức tiêu thụ > 0
    # client_1 → 0.95  (95% thời gian có điện)
    # client_2 → 0.10  (chủ yếu = 0 → rác)
    active_ratio = (
        (df_hourly > 0)
        .mean()
    )

    # tính mức tiêu thụ trung bìn từng user
    # client_1 → 3.2 kW
    # client_2 → 0.1 kW
    mean_load = (
        df_hourly.mean()
    )

    # gom lại thành bảng stats
    client_stats = pd.DataFrame({
        "active_ratio": active_ratio,
        "mean_load": mean_load
    })


    client_stats = client_stats[
        client_stats["mean_load"]
        >= min_mean_load
    ]

    keep_clients = (
        client_stats
        .sort_values(
            by="active_ratio",
            ascending=False
        )
        .head(top_clients)
        .index
        .tolist()
    )

    df_hourly = df_hourly[
        keep_clients
    ]

    # ==== save ====
    df_hourly.to_csv(
        output_path,
        index_label="timestamp",
        float_format="%.4f"
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        "Rows, Cols:",
        df_hourly.shape
    )

    print(
        "Clients kept:",
        len(keep_clients)
    )

    return df_hourly

# sau khi tối ưu dataset thì dùng class này để tiền xử lý
class Preprocessor:
    def __init__(
        self,
        dataset_path="backend/dataset/hourly_electricity_filtered.csv"
    ):
        self.df = pd.read_csv(
            dataset_path,
            parse_dates=["timestamp"]
        )

    # 1. wide -> long
    # chuyển dataset từ dạng rộng -> dài
    def from_wide_to_long(self):
        df = self.df.melt(
            id_vars="timestamp",
            var_name="client_id",
            value_name="target_value"
        )

        # bỏ zero
        df = df[
            df["target_value"] != 0
        ].copy()


        df["client_id"] = (
            df["client_id"]
            .astype("category")
        )

        df["target_value"] = (
            df["target_value"]
            .astype("float32")
        )

        print(
            "after melt:",
            df.shape
        )

        return df

    # 2. time features
    # thêm các đặc trưng về thời gian vào dataset
    def build_time_features(self,
        df
    ):
        df = df.sort_values(
            ["client_id","timestamp"]
        )

        df["day_of_week"] = (
            df["timestamp"]
            .dt.dayofweek
            .astype("int8")
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype("int8")

        hour = (
            df["timestamp"]
            .dt.hour
        )

        df["hour_sin"] = np.sin(
            2*np.pi*hour/24
        ).astype(
            "float32"
        )

        df["hour_cos"] = np.cos(
            2*np.pi*hour/24
        ).astype(
            "float32"
        )

        return df

    # 3. lag features
    # thêm các đặc trựng về lịch sử 
    def build_lag_features(self,
        df
    ):
        g = df.groupby(
            "client_id",
            observed=True
        )["target_value"]

        df["lag_24h"] = (
            g.shift(24)
            .astype("float32")
        )

        df["lag_7d"] = (
            g.shift(168)
            .astype("float32")
        )

        df["rolling_mean_24h"] = (
            g.shift(1)
             .rolling(
                24,
                min_periods=24
             )
             .mean()
             .astype(
                "float32"
             )
        )

        df.dropna(
            inplace=True
        )

        print(
            "after lag:",
            df.shape
        )

        return df

    # 4. target transform
    def transform_target(
        self,
        df
    ):
        df["target_value"] = (
            np.log1p(
                df["target_value"]
            )
            .astype(
                "float32"
            )
        )

        return df

    # 5. split time series
    def split_data(self,
        df,
        train_end="2014-09-30",
        valid_end="2014-11-30"
    ):
        train = df[
            df["timestamp"]
            <= train_end
        ].copy()

        valid = df[
            (
                df["timestamp"] > train_end
            )
            &
            (
                df["timestamp"] <= valid_end
            )
        ].copy()

        test = df[
            df["timestamp"]
            > valid_end
        ].copy()

        print("train:", train.shape)
        print("valid:", valid.shape)
        print("test:", test.shape)

        return (train, valid, test)

    # 6 save split files
    def save_splits(self,
        train,
        valid,
        test,
        base_path="backend/dataset/"
    ):
        train.to_csv(
            f"{base_path}train_ready.csv",
            index=False
        )

        valid.to_csv(
            f"{base_path}valid_ready.csv",
            index=False
        )

        test.to_csv(
            f"{base_path}test_ready.csv",
            index=False
        )

        print("saved split files")

    # 7. load split files
    @staticmethod
    def load_train_data():
        return pd.read_csv(
            "backend/dataset/train_ready.csv",
            parse_dates=["timestamp"]
        )

    @staticmethod
    def load_valid_data():
        return pd.read_csv(
            "backend/dataset/valid_ready.csv",
            parse_dates=["timestamp"]
        )

    @staticmethod
    def load_test_data():
        return pd.read_csv(
            "backend/dataset/test_ready.csv",
            parse_dates=["timestamp"]
        )

    # model columns
    @staticmethod
    def feature_columns():
        return [
            "hour_sin",
            "hour_cos",
            "day_of_week",

            "lag_24h",
            "lag_7d",

            "rolling_mean_24h"
        ]

    # full pipeline
    def run_pipeline(self):
        df = self.from_wide_to_long()
        df = self.build_time_features(df)
        df = self.build_lag_features(df)
        df = self.transform_target(df)

        print("final:", df.shape)

        train, valid, test = self.split_data(df)
        self.save_splits(train, valid, test)

        return ( train, valid, test )