import pandas as pd

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
