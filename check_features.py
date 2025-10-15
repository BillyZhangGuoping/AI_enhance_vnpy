import pandas as pd
import os

# 读取 Parquet 文件
def load_and_check_features():
    data_path = os.path.join("processed_data", "processed_data.parquet")
    try:
        df = pd.read_parquet(data_path)
        print(f"文件中的特征数量: {len(df.columns)}")
        print("特征列名:", df.columns.tolist())
    except Exception as e:
        print(f"读取文件失败: {e}")

if __name__ == "__main__":
    load_and_check_features()