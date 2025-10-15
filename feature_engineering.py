import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import os
import pickle

# 1. 读取 Parquet 数据
def load_data():
    data_path = os.path.join("processed_data", "fu2601_1M_processed_data.parquet")
    df = pd.read_parquet(data_path)
    return df

# 2. 特征标准化
def standardize_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# 3. 特征选择（随机森林）
def select_features(df, target_col, n_features=15):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    selector = SelectFromModel(model, max_features=n_features, threshold=-np.inf)
    selector.fit(X, y)
    selected_cols = X.columns[selector.get_support()]
    
    return df[selected_cols]

# 4. 处理多重共线性
def remove_collinear_features(df, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df

# 5. 创建滞后特征
def create_lag_features(df, lag_periods=3):
    lagged_df = df.copy()
    for col in df.columns:
        for lag in range(1, lag_periods + 1):
            lagged_df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return lagged_df.dropna()

# 主函数
def main():
    # 1. 加载数据
    df = load_data()
    
    # 确认目标列名
    target_col = "future_close"  # 根据数据列名调整
    
    # 动态排除非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"目标列 '{target_col}' 不是数值类型")
    
    # 确保仅使用数值列
    df = df[numeric_cols]
    
    # 处理缺失值
    df = df.ffill().dropna()  # 使用 ffill() 替代 fillna(method='ffill')
    if df.empty:
        raise ValueError("数据为空，请检查缺失值处理")
    print(f"处理后的数据行数: {len(df)}")
    
    # 2. 标准化
    df = standardize_features(df, numeric_cols)
    
    # 3. 特征选择
    selected_df = select_features(df, target_col)
    
    # 4. 处理多重共线性
    final_df = remove_collinear_features(selected_df)
    
    # 5. 创建滞后特征
    lagged_df = create_lag_features(final_df)
    
    # 6. 保存结果
    output_dir = "features"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "engineered_features.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(lagged_df, f)
    
    print(f"处理后的特征集已保存至 {output_path}")

if __name__ == "__main__":
    main()