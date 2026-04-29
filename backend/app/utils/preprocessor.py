import pandas as pd
import numpy as np

# class kiểm tra dữ liệu
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

# tạo dataset hourly và lưu ra csv
def build_hourly_dataset(
    input_path="backend/dataset/LD2011_2014.txt",
    output_path="backend/dataset/hourly_electricity_filtered.csv",

    top_clients=180,
    min_mean_load=1.0,

    # chỉ giữ 1 năm cuối
    keep_year="2014"
):
    df = pd.read_csv(
        input_path,
        sep=";",
        decimal=",",
        parse_dates=[0]
    )


    df.rename(
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


    # ====================
    # 15 phút -> 1 giờ
    # ====================
    df_hourly = (
        df
        .resample("1h")
        .sum()
    )


    # ====================
    # chỉ giữ 1 năm
    # ====================
    df_hourly = df_hourly.loc[
        keep_year
    ]


    print(
        "after year filter:",
        df_hourly.shape
    )


    # ====================
    # chọn active clients
    # ====================
    active_ratio = (
        (df_hourly > 0)
        .mean()
    )

    mean_load = (
        df_hourly.mean()
    )

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


    # ====================
    # save
    # ====================
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

import numpy as np
import pandas as pd


class Preprocessor:

    def __init__(
        self,
        dataset_path="backend/dataset/hourly_electricity_filtered.csv"
    ):
        self.df = pd.read_csv(
            dataset_path,
            parse_dates=["timestamp"]
        )


    # =================================
    # 1 wide -> long
    # =================================
    def from_wide_to_long(
        self
    ):

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


    # =================================
    # 2 time features
    # =================================
    def build_time_features(
        self,
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



    # =================================
    # 3 lag features
    # =================================
    def build_lag_features(
        self,
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



    # =================================
    # 4 target transform
    # =================================
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



    # =================================
    # 5 split time series
    # =================================
    def split_data(
        self,
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


        print(
            "train:",
            train.shape
        )

        print(
            "valid:",
            valid.shape
        )

        print(
            "test:",
            test.shape
        )


        return (
            train,
            valid,
            test
        )



    # =================================
    # 6 save split files
    # =================================
    def save_splits(
        self,
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


        print(
            "saved split files"
        )



    # =================================
    # load split files
    # =================================
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



    # =================================
    # model columns
    # =================================
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



    # =================================
    # full pipeline
    # =================================
    def run_pipeline(
        self
    ):

        df = self.from_wide_to_long()

        df = self.build_time_features(
            df
        )

        df = self.build_lag_features(
            df
        )

        df = self.transform_target(
            df
        )


        print(
            "final:",
            df.shape
        )


        train,valid,test = (
            self.split_data(
                df
            )
        )


        self.save_splits(
            train,
            valid,
            test
        )


        return (
            train,
            valid,
            test
        )

class DataTransformer:
    def __init__(
        self,
        target_col="target_value"
    ):
        self.target_col = target_col


    # ---------------------------------
    # log transform target
    # ---------------------------------
    def log_transform_target(
        self,
        df
    ):
        df = df.copy()

        df[self.target_col] = (
            np.log1p(
                df[self.target_col]
            )
            .astype("float32")
        )

        print(
            "Applied log1p transform"
        )

        return df


    # ---------------------------------
    # inverse transform prediction
    # ---------------------------------
    def inverse_transform_prediction(
        self,
        preds
    ):
        return np.expm1(preds)


    # ---------------------------------
    # train valid test split
    # time series split
    # ---------------------------------
    def train_valid_test_split(
        self,
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


        print(
            "Train:",
            train.shape
        )

        print(
            "Valid:",
            valid.shape
        )

        print(
            "Test:",
            test.shape
        )

        return (
            train,
            valid,
            test
        )


    # ---------------------------------
    # features for model
    # ---------------------------------
    def get_feature_columns(self):

        return [
            "hour_sin",
            "hour_cos",
            "day_of_week",

            "lag_24h",
            "lag_7d",

            "rolling_mean_24h"
        ]


    # ---------------------------------
    # X y extraction
    # ---------------------------------
    def build_model_inputs(
        self,
        df
    ):
        features = (
            self.get_feature_columns()
        )

        X = df[
            features
        ]

        y = df[
            self.target_col
        ]

        return (
            X,
            y
        )


    # ---------------------------------
    # full pipeline
    # ---------------------------------
    def run_pipeline(
        self,
        df
    ):

        df = self.log_transform_target(
            df
        )

        (
            train_df,
            valid_df,
            test_df
        ) = self.train_valid_test_split(
            df
        )


        X_train,y_train = (
            self.build_model_inputs(
                train_df
            )
        )

        X_valid,y_valid = (
            self.build_model_inputs(
                valid_df
            )
        )

        X_test,y_test = (
            self.build_model_inputs(
                test_df
            )
        )


        return {
            "X_train": X_train,
            "y_train": y_train,

            "X_valid": X_valid,
            "y_valid": y_valid,

            "X_test": X_test,
            "y_test": y_test
        }


#     # ------------------------
#     # 5. SPLIT TIME-BASED
#     # ------------------------
#     def split(self):

#         df = self.df.sort_values("time")

#         split = int(len(df) * (1 - self.test_size))

#         train = df.iloc[:split]
#         test = df.iloc[split:]

#         X_train = train[self.features]
#         y_train = train["load"]

#         X_test = test[self.features]
#         y_test = test["load"]

#         return X_train, X_test, y_train, y_test


#     # ------------------------
#     # 6. TRAIN
#     # ------------------------
#     def train(self):

#         X_train, X_test, y_train, y_test = self.split()

#         self.model.fit(X_train, y_train)

#         self.X_test = X_test
#         self.y_test = y_test
#         self.y_pred = self.model.predict(X_test)

#         return self


#     # ------------------------
#     # 7. EVALUATE
#     # ------------------------
#     def evaluate(self):

#         mae = mean_absolute_error(self.y_test, self.y_pred)
#         rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))

#         return {
#             "MAE": float(mae),
#             "RMSE": float(rmse)
#         }


#     # ------------------------
#     # FULL PIPELINE
#     # ------------------------
#     def run(self):

#         print("Loading data...")
#         self.load_data()

#         print("Converting to long format...")
#         self.to_long()

#         print("Trimming structural zeros...")
#         self.trim_clients()

#         print("Resampling to 1H...")
#         self.resample_1h()

#         print("Building features...")
#         self.build_features()

#         print("Training model...")
#         self.train()

#         print("Evaluating...")
#         return self.evaluate()

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error

class GlobalElectricityModel:
    """
    FIXED GLOBAL LIGHTGBM PIPELINE
    - log transform để ổn định variance
    - clip outlier 3-sigma per client
    - xử lý zero block dài giữa chuỗi
    - split đúng theo cutoff thời gian
    - debug prints đầy đủ ở mỗi bước
    """

    def __init__(
        self,
        dataset_path,
        test_size=0.2,
        use_log=True,
        zero_block_threshold=672,   # 7 ngày * 96 records/ngày (15 phút)
        outlier_sigma=3.0
    ):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.use_log = use_log
        self.zero_block_threshold = zero_block_threshold
        self.outlier_sigma = outlier_sigma

        self.raw_df = None
        self.df = None

        self.model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            random_state=42
        )

        self.features = [
            "lag_1", "lag_24", "lag_48", "lag_168",
            "rolling_24",
            "hour", "day_of_week", "month",
            "client_code"
        ]


    # ----------------------
    # LOAD DATA
    # ----------------------
    def load_data(self):
        df = pd.read_csv(
            self.dataset_path,
            sep=";",
            decimal=",",
            parse_dates=[0]
        )
        df = df.rename(columns={"Unnamed: 0": "time"})
        df = df.set_index("time")

        self.raw_df = df

        print(f"[LOAD] Shape: {df.shape}")
        print(f"[LOAD] Time range: {df.index.min()} -> {df.index.max()}")
        print(f"[LOAD] Số client: {df.shape[1]}")
        print(f"[LOAD] Missing values tổng: {df.isna().sum().sum()}")
        print(f"[LOAD] Zero values tổng: {(df == 0).sum().sum()}")
        print(f"[LOAD] % zero: {(df == 0).sum().sum() / df.size * 100:.2f}%\n")

        return df


    # ----------------------
    # WIDE -> LONG
    # ----------------------
    def to_long(self):
        df = self.raw_df.reset_index().melt(
            id_vars="time",
            var_name="client_id",
            value_name="load"
        )
        self.df = df

        print(f"[LONG] Rows: {len(df):,}")
        print(f"[LONG] Clients: {df['client_id'].nunique()}")
        print(f"[LONG] Load stats:\n{df['load'].describe()}\n")

        return df


    # ----------------------
    # REMOVE STRUCTURAL ZEROS
    # ----------------------
    def trim_clients(self):
        print(f"[TRIM] Rows trước khi trim: {len(self.df):,}")

        rows_before = len(self.df)

        def trim(group):
            # 1. Cắt đầu: bỏ toàn bộ phần leading zeros
            nonzero_idx = group["load"].ne(0).idxmax()
            group = group.loc[nonzero_idx:]

            # 2. Đánh dấu zero block dài ở giữa thành NaN
            is_zero = (group["load"] == 0)
            block_id = (is_zero != is_zero.shift()).cumsum()
            block_sizes = is_zero.groupby(block_id).transform("sum")
            long_zero_mask = is_zero & (block_sizes > self.zero_block_threshold)
            group = group.copy()
            group.loc[long_zero_mask, "load"] = np.nan

            return group

        self.df = (
            self.df
            .groupby("client_id", group_keys=False)
            .apply(trim, include_groups=False)
        )

        # Re-attach client_id nếu bị drop do include_groups=False
        if "client_id" not in self.df.columns:
            self.df = self.df.reset_index(level=0).rename(
                columns={"level_0": "client_id"}
            )

        rows_after = len(self.df)
        nan_after = self.df["load"].isna().sum()
        zero_after = (self.df["load"] == 0).sum()

        print(f"[TRIM] Rows bị cắt (leading zeros): {rows_before - rows_after:,}")
        print(f"[TRIM] Zero blocks dài -> NaN: {nan_after:,} records")
        print(f"[TRIM] Zero còn lại sau trim: {zero_after:,}")
        print(f"[TRIM] Rows sau trim: {rows_after:,}\n")

        return self.df


    # ----------------------
    # RESAMPLE 1H
    # ----------------------
    def resample(self):
        print(f"[RESAMPLE] Rows trước: {len(self.df):,}")

        df = self.df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        df = (
            df.groupby("client_id")
              .resample("1h")
              .mean()
              .reset_index()
        )

        self.df = df

        print(f"[RESAMPLE] Rows sau: {len(df):,}")
        print(f"[RESAMPLE] NaN sau resample: {df['load'].isna().sum():,}")
        print(f"[RESAMPLE] Load stats sau resample:\n{df['load'].describe()}\n")

        return df


    # ----------------------
    # FEATURE ENGINEERING
    # ----------------------
    def build_features(self):
        df = self.df.copy()
        df = df.sort_values(["client_id", "time"])

        print(f"[FEATURES] Rows trước khi build: {len(df):,}")
        print(f"[FEATURES] Load raw — max: {df['load'].max():.4f}, "
              f"p99: {df['load'].quantile(0.99):.4f}, "
              f"p99.9: {df['load'].quantile(0.999):.4f}")

        # LOG TRANSFORM
        if self.use_log:
            df["load"] = np.log1p(df["load"])
            print(f"[FEATURES] Sau log1p — max: {df['load'].max():.4f}, "
                  f"mean: {df['load'].mean():.4f}, "
                  f"std: {df['load'].std():.4f}")

        # CLIP OUTLIER 3-SIGMA PER CLIENT
        before_clip = df["load"].describe()

        def clip_outliers(x):
            mu = x.mean()
            sigma = x.std()
            return x.clip(mu - self.outlier_sigma * sigma,
                          mu + self.outlier_sigma * sigma)

        df["load"] = (
            df.groupby("client_id")["load"]
              .transform(clip_outliers)
        )

        after_clip = df["load"].describe()
        print(f"[FEATURES] Sau clip {self.outlier_sigma}-sigma — "
              f"max: {df['load'].max():.4f}, "
              f"min: {df['load'].min():.4f}")
        print(f"[FEATURES] Delta max do clip: "
              f"{before_clip['max'] - after_clip['max']:.4f}")

        # LAG FEATURES
        for lag in [1, 24, 48, 168]:
            df[f"lag_{lag}"] = (
                df.groupby("client_id")["load"]
                  .shift(lag)
            )

        # ROLLING 24H (SAFE — shift(1) trước để tránh leakage)
        df["rolling_24"] = (
            df.groupby("client_id")["load"]
              .transform(lambda x: x.shift(1).rolling(24).mean())
        )

        # TIME FEATURES
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["month"] = df["time"].dt.month

        # CLIENT ENCODING
        df["client_code"] = df["client_id"].astype("category").cat.codes

        rows_before_drop = len(df)
        df = df.dropna()
        rows_dropped = rows_before_drop - len(df)

        print(f"[FEATURES] Rows bị dropna: {rows_dropped:,}")
        print(f"[FEATURES] Rows cuối cùng: {len(df):,}")
        print(f"[FEATURES] Load final stats:\n{df['load'].describe()}\n")

        self.df = df
        return df


    # ----------------------
    # SPLIT (THEO THỜI GIAN)
    # ----------------------
    def split(self):
        df = self.df.sort_values("time")

        # Cutoff theo quantile thời gian, không theo row index
        cutoff = df["time"].quantile(1 - self.test_size)
        if not isinstance(cutoff, pd.Timestamp):
            cutoff = pd.Timestamp(cutoff)

        train = df[df["time"] < cutoff]
        test  = df[df["time"] >= cutoff]

        print(f"[SPLIT] Cutoff: {cutoff}")
        print(f"[SPLIT] Train rows: {len(train):,} | "
              f"Train time: {train['time'].min()} -> {train['time'].max()}")
        print(f"[SPLIT] Test  rows: {len(test):,}  | "
              f"Test  time: {test['time'].min()} -> {test['time'].max()}")
        print(f"[SPLIT] Train clients: {train['client_id'].nunique()} | "
              f"Test clients: {test['client_id'].nunique()}\n")

        X_train = train[self.features]
        y_train = train["load"]
        X_test  = test[self.features]
        y_test  = test["load"]

        return X_train, X_test, y_train, y_test


    # ----------------------
    # TRAIN
    # ----------------------
    def train(self):
        X_train, X_test, y_train, y_test = self.split()

        print(f"[TRAIN] Bắt đầu fit model...")
        self.model.fit(X_train, y_train)
        print(f"[TRAIN] Fit xong.")

        self.X_test  = X_test
        self.y_test  = y_test
        self.y_pred  = self.model.predict(X_test)

        print(f"[TRAIN] y_pred (log scale) — "
              f"min: {self.y_pred.min():.4f}, "
              f"max: {self.y_pred.max():.4f}, "
              f"mean: {self.y_pred.mean():.4f}")
        print(f"[TRAIN] y_test (log scale) — "
              f"min: {float(self.y_test.min()):.4f}, "
              f"max: {float(self.y_test.max()):.4f}, "
              f"mean: {float(self.y_test.mean()):.4f}\n")

        return self


    # ----------------------
    # EVALUATE (INVERSE LOG)
    # ----------------------
    def evaluate(self):
        y_true = self.y_test.copy()
        y_pred = self.y_pred.copy()

        if self.use_log:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)

        residuals = np.abs(y_true.values - y_pred)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"[EVAL] MAE : {mae:.4f}")
        print(f"[EVAL] RMSE: {rmse:.4f}")
        print(f"[EVAL] --- Phân tích residuals ---")
        print(f"[EVAL] Max error    : {residuals.max():.2f}")
        print(f"[EVAL] p95 error    : {np.percentile(residuals, 95):.2f}")
        print(f"[EVAL] p99 error    : {np.percentile(residuals, 99):.2f}")
        print(f"[EVAL] p99.9 error  : {np.percentile(residuals, 99.9):.2f}")
        print(f"[EVAL] Errors > 1000: {(residuals > 1000).sum():,} "
              f"({(residuals > 1000).mean()*100:.3f}%)")
        print(f"[EVAL] Errors > 500 : {(residuals > 500).sum():,} "
              f"({(residuals > 500).mean()*100:.3f}%)")
        print(f"[EVAL] Errors > 100 : {(residuals > 100).sum():,} "
              f"({(residuals > 100).mean()*100:.3f}%)")

        # Feature importance top 9
        fi = dict(zip(self.features, self.model.feature_importances_))
        fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
        print(f"\n[EVAL] Feature importance:")
        for feat, imp in fi_sorted:
            print(f"       {feat:<15} {imp}")

        return {
            "MAE": float(mae),
            "RMSE": float(rmse)
        }


    # ----------------------
    # RUN PIPELINE
    # ----------------------
    def run(self):
        print("="*55)
        print("  GLOBAL ELECTRICITY MODEL — BẮT ĐẦU PIPELINE")
        print("="*55 + "\n")

        print(">>> [1/6] Loading data...")
        self.load_data()

        print(">>> [2/6] Wide -> Long format...")
        self.to_long()

        print(">>> [3/6] Trimming zeros...")
        self.trim_clients()

        print(">>> [4/6] Resampling 1h...")
        self.resample()

        print(">>> [5/6] Building features...")
        self.build_features()

        print(">>> [6/6] Training & Evaluating...")
        self.train()

        print("\n" + "="*55)
        print("  KẾT QUẢ CUỐI CÙNG")
        print("="*55)
        result = self.evaluate()

        return result
if __name__ == "__main__":
    preprocessor = Preprocessor()
    # preprocessor.check_zero_user()
    print(preprocessor.max_consecutive_zeros(preprocessor.df['MT_001']))