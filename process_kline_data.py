import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from datetime import datetime
import sqlite3
import os

# 1. 从CSV文件加载1分钟K线数据
def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    从CSV文件加载K线数据
    :param file_path: CSV文件路径
    :return: DataFrame包含K线数据
    """
    df = pd.read_csv(file_path)
    return df

# 2. 数据清洗
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：处理缺失值和异常值
    :param df: 原始数据
    :return: 清洗后的数据
    """
    # 处理缺失值（线性插值）
    df = df.infer_objects(copy=False)
    df = df.interpolate(method='linear')
    
    # 去除异常值（3σ原则）
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] > mean - 3 * std) & (df[col] < mean + 3 * std)]
    return df

# 3. 添加技术指标
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加技术指标
    :param df: 清洗后的数据
    :return: 包含技术指标的数据
    """
    # 收盘价的均值
    df['close_5ma'] = df['close'].rolling(window=5).mean()
    df['close_10ma'] = df['close'].rolling(window=10).mean()
    df['close_20ma'] = df['close'].rolling(window=20).mean()
    df['close_30ma'] = df['close'].rolling(window=30).mean()
    
    # 动量指标
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # 波动率指标
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    # 成交量指标
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14).volume_weighted_average_price()
    
    return df

# 4. 添加时间特征
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加时间特征
    :param df: 包含技术指标的数据
    :return: 包含时间特征的数据
    """
    df['bob'] = pd.to_datetime(df['bob'])
    df['hour'] = df['bob'].dt.hour
    df['minute'] = df['bob'].dt.minute
    df['day_of_week'] = df['bob'].dt.dayofweek
    return df

# 5. 定义目标变量
def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    定义目标变量：未来5根K线的价格变化方向
    :param df: 包含时间特征的数据
    :return: 包含目标变量的数据
    """
    df['future_close'] = df['close'].shift(-5)
    price_change = df['future_close'] - df['close']
    
    # 最小变动单位为1
    min_change = 1
    
    # 判断涨跌
    df['target'] = 0  # 默认无序波动
    df.loc[price_change > min_change, 'target'] = 1  # 涨
    df.loc[price_change < -min_change, 'target'] = -1  # 跌
    
    df = df.dropna(subset=['target'])
    return df

# 6. 保存为Parquet文件
def save_to_parquet(df: pd.DataFrame, input_path: str):
    """
    保存为Parquet文件
    :param df: 最终处理完成的数据
    :param input_path: 输入文件路径
    """
    # 提取文件名（不带扩展名）
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"./data/{file_name}_processed_data.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

# 7. 保存为CSV文件
def save_to_csv(df: pd.DataFrame, input_path: str):
    """
    保存为CSV文件
    :param df: 最终处理完成的数据
    :param input_path: 输入文件路径
    """
    # 提取文件名（不带扩展名）
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"./processed_data/{file_name}_processed_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

# 主函数
def main():
    # CSV文件路径
    csv_path = "./future_data/fu2601_1M.csv"
    
    # 加载数据
    df = load_data_from_csv(csv_path)
    
    # 数据清洗
    df = clean_data(df)
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 添加时间特征
    df = add_time_features(df)
    
    # 定义目标变量
    df = add_target_variable(df)
    
    # 保存为Parquet文件
    save_to_parquet(df, csv_path)
    print(f"数据已保存到 ./data/{os.path.splitext(os.path.basename(csv_path))[0]}_processed_data.parquet")
    
    # 保存为CSV文件
    save_to_csv(df, csv_path)
    print(f"数据已保存到 ./data/{os.path.splitext(os.path.basename(csv_path))[0]}_processed_data.csv")

if __name__ == "__main__":
    main()