import numpy as np
import pandas as pd
import torch
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime

# 确保中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建目录
os.makedirs('./plots', exist_ok=True)
os.makedirs('./validation_results', exist_ok=True)

class LSTMTradingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(LSTMTradingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
        # Dropout层
        self.dropout = torch.nn.Dropout(dropout)
        
        # Batch Normalization
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # Batch Normalization
        out = self.bn(out)
        
        # Dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        
        return out

def load_data(file_path):
    """加载期货数据"""
    print(f"加载数据: {file_path}")
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['bob'].min()} 到 {df['bob'].max()}")
    return df

def preprocess_data(df):
    """数据预处理 - 与训练时保持一致"""
    # 复制数据以避免修改原始数据
    df_processed = df.copy()
    
    # 计算基本价格特征
    df_processed['high_low_ratio'] = df_processed['high'] / df_processed['low']
    df_processed['open_close_diff'] = df_processed['open'] - df_processed['close']
    df_processed['price_range'] = df_processed['high'] - df_processed['low']
    
    # 计算收益率
    for i in [1, 3, 5, 10]:
        df_processed[f'returns_{i}'] = df_processed['close'].pct_change(i)
    
    # 计算移动平均线
    for i in [5, 10, 20, 50]:
        df_processed[f'ma_{i}'] = df_processed['close'].rolling(window=i).mean()
        df_processed[f'ma_diff_{i}'] = df_processed['close'] - df_processed[f'ma_{i}']
    
    # 计算相对强弱指数(RSI)
    delta = df_processed['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_processed['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_processed['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_processed['close'].ewm(span=26, adjust=False).mean()
    df_processed['macd'] = exp1 - exp2
    df_processed['signal_line'] = df_processed['macd'].ewm(span=9, adjust=False).mean()
    df_processed['macd_diff'] = df_processed['macd'] - df_processed['signal_line']
    
    # 交易量特征
    df_processed['volume_pct_change'] = df_processed['volume'].pct_change()
    df_processed['volume_ma_10'] = df_processed['volume'].rolling(window=10).mean()
    df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume_ma_10']
    
    # 持仓量特征
    df_processed['position_pct_change'] = df_processed['position'].pct_change()
    
    # 创建目标变量 - 使用未来5分钟的价格变化作为预测目标
    df_processed['future_returns'] = df_processed['close'].pct_change(5).shift(-5)
    df_processed['target'] = 0
    df_processed.loc[df_processed['future_returns'] > 0.0005, 'target'] = 1  # 多单
    df_processed.loc[df_processed['future_returns'] < -0.0005, 'target'] = -1  # 空单
    
    # 填充缺失值 - 只对数值列应用中位数填充
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    return df_processed

def create_sequences(data, seq_length, feature_columns, target_column):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[feature_columns].iloc[i:i+seq_length].values)
        y.append(data[target_column].iloc[i+seq_length])
    return np.array(X), np.array(y)

def load_model(model_path, device, input_dim=11, hidden_dim=128, num_layers=3, output_dim=3, dropout=0.4):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    # 创建模型实例
    model = LSTMTradingModel(input_dim, hidden_dim, num_layers, output_dim, dropout).to(device)
    # 加载模型状态字典
    state_dict = torch.load(model_path, map_location=device)
    # 如果直接是状态字典，直接加载
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        # 否则尝试直接使用（可能是完整模型）
        model = state_dict
    model.eval()
    return model

def load_scaler(scaler_path):
    """加载特征缩放器"""
    print(f"加载缩放器: {scaler_path}")
    try:
        # 尝试使用joblib加载（通常训练脚本使用joblib）
        scaler = joblib.load(scaler_path)
    except:
        try:
            # 尝试使用pickle加载
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except:
            # 如果都失败，创建一个新的缩放器
            print("无法加载缩放器，创建新的缩放器")
            scaler = StandardScaler()
    return scaler

def validate_model(model, X, y, scaler, device, seq_length):
    """验证模型性能"""
    # 对特征进行缩放
    X_flat = X.reshape(-1, X.shape[2])
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(-1, seq_length, X.shape[2])
    
    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # 转换预测结果
    predicted_labels = np.array([-1, 0, 1])[predicted.cpu().numpy()]
    
    # 计算整体准确率
    overall_accuracy = np.mean(predicted_labels == y)
    
    # 计算交易信号准确率（只考虑非零信号）
    trade_mask = (y != 0) & (predicted_labels != 0)
    if np.sum(trade_mask) > 0:
        trade_accuracy = np.mean(predicted_labels[trade_mask] == y[trade_mask])
    else:
        trade_accuracy = 0
    
    # 计算多单信号准确率
    long_mask = (y == 1) & (predicted_labels != 0)
    if np.sum(long_mask) > 0:
        long_accuracy = np.mean(predicted_labels[long_mask] == y[long_mask])
    else:
        long_accuracy = 0
    
    # 计算空单信号准确率
    short_mask = (y == -1) & (predicted_labels != 0)
    if np.sum(short_mask) > 0:
        short_accuracy = np.mean(predicted_labels[short_mask] == y[short_mask])
    else:
        short_accuracy = 0
    
    # 准备用于分类报告的数据 - 转换为0,1,2标签
    y_encoded = np.zeros_like(y)
    y_encoded[y == -1] = 0
    y_encoded[y == 1] = 1
    
    predicted_encoded = np.zeros_like(predicted_labels)
    predicted_encoded[predicted_labels == -1] = 0
    predicted_encoded[predicted_labels == 1] = 1
    
    # 生成混淆矩阵
    cm = confusion_matrix(y, predicted_labels, labels=[-1, 0, 1])
    
    return {
        'overall_accuracy': overall_accuracy,
        'trade_accuracy': trade_accuracy,
        'long_accuracy': long_accuracy,
        'short_accuracy': short_accuracy,
        'y_true': y,
        'y_pred': predicted_labels,
        'y_true_encoded': y_encoded,
        'y_pred_encoded': predicted_encoded,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, title, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    # 确保只使用交易信号类别(-1,1)的混淆矩阵部分
    cm_trade = cm[[0, 2]][:, [0, 2]]
    
    sns.heatmap(cm_trade, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['空单预测', '多单预测'],
                yticklabels=['空单实际', '多单实际'])
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_performance_comparison(results, save_path):
    """绘制不同数据集上的性能对比"""
    datasets = list(results.keys())
    metrics = ['overall_accuracy', 'trade_accuracy', 'long_accuracy', 'short_accuracy']
    metric_names = ['整体准确率', '交易信号准确率', '多单准确率', '空单准确率']
    
    # 准备数据
    data = {}
    for metric in metrics:
        data[metric] = [results[dataset][metric] for dataset in datasets]
    
    # 绘制条形图
    fig, ax = plt.subplots(figsize=(15, 10))
    width = 0.2
    x = np.arange(len(datasets))
    
    for i, (metric, values) in enumerate(data.items()):
        ax.bar(x + i * width, values, width, label=metric_names[i])
    
    ax.set_xlabel('数据集')
    ax.set_ylabel('准确率')
    ax.set_title('不同数据集上的模型性能对比')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for i, (metric, values) in enumerate(data.items()):
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # 配置参数
    config = {
        'sequence_length': 10,
        'hidden_dim': 128,
        'num_layers': 3,
        'output_dim': 3,  # -1, 0, 1
        'dropout': 0.4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"使用设备: {config['device']}")
    
    # 数据文件路径
    data_files = {
        'fu2505': 'C:\\python_workspace\\future_data\\fu2505_1M.csv',
        'fu2510': 'C:\\python_workspace\\future_data\\fu2510_1M.csv'
    }
    
    # 模型和缩放器路径
    model_path = './models/3_layers_128hd_0.4do_model.pkl'
    scaler_path = './models/experiment_scaler.pkl'
    
    # 特征列 - 与训练时保持一致（22个特征）
    feature_columns = [
        'high_low_ratio', 'open_close_diff', 'price_range',
        'returns_1', 'returns_3', 'returns_5', 'returns_10',
        'ma_5', 'ma_diff_5', 'ma_10', 'ma_diff_10', 'ma_20', 'ma_diff_20', 'ma_50',
        'rsi', 'macd', 'signal_line', 'macd_diff',
        'volume_pct_change', 'volume_ma_10', 'volume_ratio', 'position_pct_change'
    ]  # 移除了'ma_diff_50'，现在是22个特征
    target_column = 'target'
    
    # 加载模型和缩放器
    # 计算输入维度（特征数量）
    input_dim = len(feature_columns)
    model = load_model(model_path, config['device'], 
                      input_dim=input_dim,
                      hidden_dim=config['hidden_dim'],
                      num_layers=config['num_layers'],
                      output_dim=config['output_dim'],
                      dropout=config['dropout'])
    scaler = load_scaler(scaler_path)
    
    # 验证结果
    validation_results = {}
    
    # 对每个数据集进行验证
    for dataset_name, file_path in data_files.items():
        print(f"\n处理数据集: {dataset_name}")
        
        # 加载和预处理数据
        df = load_data(file_path)
        df_processed = preprocess_data(df)
        
        # 创建序列
        X, y = create_sequences(df_processed, config['sequence_length'], feature_columns, target_column)
        print(f"创建序列: {X.shape}, {y.shape}")
        
        # 验证模型
        results = validate_model(model, X, y, scaler, config['device'], config['sequence_length'])
        validation_results[dataset_name] = results
        
        # 打印结果
        print(f"\n===== 验证结果: {dataset_name} =====")
        print(f"整体准确率: {results['overall_accuracy']:.4f}")
        print(f"交易信号准确率: {results['trade_accuracy']:.4f}")
        print(f"多单准确率: {results['long_accuracy']:.4f}")
        print(f"空单准确率: {results['short_accuracy']:.4f}")
        
        # 生成分类报告
        print("\n分类报告:")
        print(classification_report(results['y_true_encoded'], results['y_pred_encoded'], 
                                    target_names=['空单(-1)', '多单(1)']))
        
        # 绘制混淆矩阵
        cm_path = f"./validation_results/{dataset_name}_confusion_matrix.png"
        plot_confusion_matrix(results['confusion_matrix'], f"{dataset_name} 混淆矩阵", cm_path)
        print(f"混淆矩阵保存至: {cm_path}")
        
        # 统计各类别的数量
        unique, counts = np.unique(y, return_counts=True)
        print("\n类别分布:")
        for u, c in zip(unique, counts):
            print(f"类别 {u}: {c} ({c/len(y)*100:.2f}%)")
    
    # 绘制性能对比图
    comparison_path = "./validation_results/performance_comparison.png"
    plot_performance_comparison(validation_results, comparison_path)
    print(f"\n性能对比图保存至: {comparison_path}")
    
    # 保存验证结果
    results_path = "./validation_results/validation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(validation_results, f)
    print(f"验证结果保存至: {results_path}")
    
    print("\n验证完成！")

if __name__ == "__main__":
    main()