import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# 定义LSTM模型
class LSTMTradingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(LSTMTradingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_dim)
    
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

# 计算目标变量
def calculate_target(df, lookahead=5):
    # 计算未来收益
    df['future_close'] = df['close'].shift(-lookahead)
    df['return'] = (df['future_close'] - df['close']) / df['close']
    
    # 根据收益生成交易信号
    # -1: 空单信号（下跌）
    # 0: 无信号（横盘）
    # 1: 多单信号（上涨）
    threshold = 0.002  # 0.2%的阈值
    df['target'] = 0
    df.loc[df['return'] > threshold, 'target'] = 1
    df.loc[df['return'] < -threshold, 'target'] = -1
    
    # 移除NaN值
    df = df.dropna()
    
    return df

# 特征工程（简化版）
def engineer_features(df):
    # 基本价格特征
    df['hl_range'] = df['high'] - df['low']
    df['oc_diff'] = df['close'] - df['open']
    
    # 简化的收益率
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    
    # 简化的移动平均线
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_diff_5'] = df['close'] - df['ma_5']
    df['ma_diff_10'] = df['close'] - df['ma_10']
    
    # 交易量特征
    df['volume_change'] = df['volume'].pct_change()
    
    # 持仓量特征
    df['position_change'] = df['position'].pct_change()
    
    # 移除NaN值
    df = df.dropna()
    
    return df

# 加载和预处理多个CSV文件（支持优化选项）
def load_and_preprocess_multi_files(data_paths, sequence_length=10, config=None):
    print(f"正在加载多个数据文件: {data_paths}")
    
    # 设置默认配置
    if config is None:
        config = {
            'enhanced_features': False,
            'dynamic_threshold': False,
            'threshold_percentile': 0.7
        }
    
    all_data = []
    
    # 加载和处理每个文件
    for file_path in data_paths:
        print(f"处理文件: {file_path}")
        df = pd.read_csv(file_path)
        
        # 根据配置选择特征工程方法
        if config.get('enhanced_features', False):
            print("使用增强特征工程...")
            df = engineer_features_enhanced(df)
        else:
            print("使用基础特征工程...")
            df = engineer_features(df)
        
        # 根据配置选择目标计算方法
        if config.get('dynamic_threshold', False):
            print("使用动态阈值计算目标...")
            df = calculate_target_dynamic(df, lookahead=5, 
                                         threshold_percentile=config.get('threshold_percentile', 0.7))
        else:
            print("使用固定阈值计算目标...")
            df = calculate_target(df)
        
        all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"合并后数据形状: {combined_df.shape}")
    
    # 准备特征和标签
    y = combined_df['target'].values
    
    # 根据是否使用增强特征选择特征列
    if config.get('enhanced_features', False):
        # 增强版特征列
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'amount', 'position',
            # 基本价格特征
            'hl_range', 'oc_diff', 'hl_ratio', 'oc_ratio',
            # 收益率特征
            'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
            # 移动平均线特征
            'ma_5', 'ma_10', 'ma_20', 'ma_50',
            'ma_diff_5', 'ma_diff_10', 'ma_diff_20', 'ma_diff_50',
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',
            # 动量特征
            'momentum_5', 'momentum_10',
            # 波动率特征
            'volatility_5', 'volatility_10',
            # 交易量特征
            'volume_change', 'volume_ma_5', 'volume_ratio',
            # 持仓量特征
            'position_change', 'position_ma_5', 'position_ratio'
        ]
    else:
        # 简化版特征列
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'amount', 'position',
            'hl_range', 'oc_diff', 'return_1', 'return_5',
            'ma_5', 'ma_10', 'ma_diff_5', 'ma_diff_10',
            'volume_change', 'position_change'
        ]
    
    # 只保留存在的特征列
    available_features = [col for col in feature_columns if col in combined_df.columns]
    X = combined_df[available_features].copy()
    
    # 处理缺失值
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建序列数据
    X_sequences, y_sequences = create_sequences(X_scaled, y, sequence_length)
    
    return X_sequences, y_sequences, scaler, available_features

# 创建序列数据
def create_sequences(X, y, sequence_length):
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)

# 标签映射和反向映射
def get_label_mapping():
    return {-1: 0, 0: 1, 1: 2}

def get_reverse_label_mapping():
    return {0: -1, 1: 0, 2: 1}

# 增强的LSTM模型类
class EnhancedLSTMTradingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(EnhancedLSTMTradingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入层的Batch Normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层 - 增加一个中间层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # ReLU激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 对每个时间步的输入进行Batch Normalization
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)  # 重塑为(batch_size*seq_len, input_dim)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, input_dim)  # 重塑回原始形状
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 第一个全连接层
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        
        # 第二个全连接层
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# 训练模型（优化版）
def train_model(X_train, y_train, X_val, y_val, input_dim, config, experiment_name):
    # 标签映射
    label_mapping = get_label_mapping()
    y_train_mapped = np.vectorize(label_mapping.get)(y_train)
    y_val_mapped = np.vectorize(label_mapping.get)(y_val)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train_mapped)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val_mapped)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # 使用加权随机采样来处理类别不平衡
    class_counts = np.bincount(y_train_mapped)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[class_id] for class_id in y_train_mapped])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 根据配置选择模型类型
    use_enhanced_model = config.get('hidden_dim', 64) >= 128  # 隐藏层维度>=128时使用增强模型
    
    if use_enhanced_model:
        print("使用增强型LSTM模型...")
        model = EnhancedLSTMTradingModel(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=3,  # 三个类别: -1, 0, 1
            dropout=config['dropout']
        )
    else:
        print("使用标准LSTM模型...")
        model = LSTMTradingModel(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=3,
            dropout=config['dropout']
        )
    
    # 使用GPU如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config['class_weights'], device=device))
    
    # 使用AdamW优化器替代Adam
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # 使用更激进的学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.3,  # 学习率衰减更快
        patience=config['patience']//2,
        min_lr=1e-7  # 更低的最小学习率
    )
    
    # 添加学习率预热
    warmup_epochs = 5
    warmup_lr = config['learning_rate'] / 10.0
    
    # 训练循环
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # 梯度裁剪参数
    max_grad_norm = 1.0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 学习率预热
        if epoch < warmup_epochs:
            # 线性预热
            lr = warmup_lr + (config['learning_rate'] - warmup_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"预热轮次 {epoch+1}/{warmup_epochs}, 学习率: {lr:.8f}")
        
        # 混合精度训练的梯度累积步数
        accumulation_steps = 2
        
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 梯度清零（每累积steps次才清零）
            if i % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch) / accumulation_steps  # 缩放损失
            
            # 反向传播
            loss.backward()
            
            # 梯度累积和更新
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 参数更新
                optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0) * accumulation_steps
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # 确保最后一个batch也更新
        if (len(train_loader) % accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度（只在预热后使用）
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"早停在第 {epoch+1} 轮")
                break
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'轮次 [{epoch+1}/{config["epochs"]}], ' \
              f'学习率: {current_lr:.8f}, ' \
              f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, ' \
              f'训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}')
    
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.2f} 秒")
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'training_time': training_time
    }
    
    return model, history

# 评估模型
def evaluate_model(model, X_test, y_test):
    # 标签映射
    label_mapping = get_label_mapping()
    reverse_mapping = get_reverse_label_mapping()
    y_test_mapped = np.vectorize(label_mapping.get)(y_test)
    
    # 转换为PyTorch张量
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 使用GPU如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_test_tensor = X_test_tensor.to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted_mapped = torch.max(outputs, 1)
    
    # 反向映射
    predicted_mapped = predicted_mapped.cpu().numpy()
    predicted = np.vectorize(reverse_mapping.get)(predicted_mapped)
    
    # 计算整体准确率
    overall_accuracy = accuracy_score(y_test, predicted)
    
    # 过滤交易信号样本
    signal_mask = (y_test == 1) | (y_test == -1)
    y_test_signal = y_test[signal_mask]
    predicted_signal = predicted[signal_mask]
    
    # 计算交易信号准确率
    signal_accuracy = accuracy_score(y_test_signal, predicted_signal)
    
    # 计算多单和空单准确率
    long_mask = (y_test_signal == 1)
    short_mask = (y_test_signal == -1)
    
    long_accuracy = accuracy_score(y_test_signal[long_mask], predicted_signal[long_mask]) if any(long_mask) else 0
    short_accuracy = accuracy_score(y_test_signal[short_mask], predicted_signal[short_mask]) if any(short_mask) else 0
    
    # 混淆矩阵
    cm = confusion_matrix(y_test_signal, predicted_signal)
    
    # 分类报告
    report = classification_report(y_test_signal, predicted_signal, labels=[-1, 1], target_names=['空单(-1)', '多单(1)'])
    
    return {
        'overall_accuracy': overall_accuracy,
        'signal_accuracy': signal_accuracy,
        'long_accuracy': long_accuracy,
        'short_accuracy': short_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': predicted
    }

# 可视化训练过程
def plot_training_history(history, experiment_name, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 损失曲线
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title(f'Training History - {experiment_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['train_accuracies'], label='Training Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史图保存至: {save_path}")
    else:
        plt.show()

# 可视化混淆矩阵
def plot_confusion_matrix(cm, experiment_name, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Trading Signal Confusion Matrix - {experiment_name}')
    plt.colorbar()
    
    classes = ['空单(-1)', '多单(1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵中添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵保存至: {save_path}")
    else:
        plt.show()

# 创建实验比较图表
def plot_experiment_comparison(results, save_path=None):
    experiment_names = list(results.keys())
    signal_accuracies = [results[exp]['evaluation']['signal_accuracy'] for exp in experiment_names]
    long_accuracies = [results[exp]['evaluation']['long_accuracy'] for exp in experiment_names]
    short_accuracies = [results[exp]['evaluation']['short_accuracy'] for exp in experiment_names]
    training_times = [results[exp]['history']['training_time'] for exp in experiment_names]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # 准确率比较
    x = np.arange(len(experiment_names))
    width = 0.25
    
    ax1.bar(x - width, signal_accuracies, width, label='Signal Accuracy')
    ax1.bar(x, long_accuracies, width, label='Long Accuracy')
    ax1.bar(x + width, short_accuracies, width, label='Short Accuracy')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance Comparison Across Experiments')
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y')
    
    # 训练时间比较
    ax2.bar(experiment_names, training_times)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"实验比较图保存至: {save_path}")
    else:
        plt.show()

# 特征工程（增强版）
def engineer_features_enhanced(df):
    # 基本价格特征
    df['hl_range'] = df['high'] - df['low']
    df['oc_diff'] = df['close'] - df['open']
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['oc_ratio'] = (df['close'] - df['open']) / df['open']
    
    # 扩展的收益率
    for n in [1, 3, 5, 10, 20]:
        df[f'return_{n}'] = df['close'].pct_change(n)
    
    # 扩展的移动平均线
    for n in [5, 10, 20, 50]:
        df[f'ma_{n}'] = df['close'].rolling(window=n).mean()
        df[f'ma_diff_{n}'] = df['close'] - df[f'ma_{n}']
        df[f'ma_ratio_{n}'] = df['close'] / df[f'ma_{n}']
    
    # 动量指标
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    
    # 波动率指标
    df['volatility_5'] = df['close'].rolling(window=5).std()
    df['volatility_10'] = df['close'].rolling(window=10).std()
    
    # 交易量特征
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # 持仓量特征
    df['position_change'] = df['position'].pct_change()
    df['position_ma_5'] = df['position'].rolling(window=5).mean()
    df['position_ratio'] = df['position'] / df['position_ma_5']
    
    # 移除NaN值
    df = df.dropna()
    
    return df

# 计算目标变量（动态阈值版）
def calculate_target_dynamic(df, lookahead=5, threshold_percentile=0.7):
    # 计算未来收益
    df['future_close'] = df['close'].shift(-lookahead)
    df['return'] = (df['future_close'] - df['close']) / df['close']
    
    # 计算动态阈值
    positive_threshold = df['return'].quantile(threshold_percentile)
    negative_threshold = df['return'].quantile(1 - threshold_percentile)
    print(f"使用动态阈值: 正阈值={positive_threshold:.6f}, 负阈值={negative_threshold:.6f}")
    
    # 根据收益生成交易信号
    df['target'] = 0
    df.loc[df['return'] > positive_threshold, 'target'] = 1
    df.loc[df['return'] < negative_threshold, 'target'] = -1
    
    # 移除NaN值
    df = df.dropna()
    
    # 打印类别分布
    print(f"目标变量分布: {df['target'].value_counts().to_dict()}")
    
    return df

# 主函数 - 训练多个LSTM模型
def main():
    print("===== LSTM交易信号预测模型 - 多文件训练 ======")
    
    # 数据路径 - 三个CSV文件
    data_paths = [
        'C:\\python_workspace\\future_data\\fu2505_1M.csv',
        'C:\\python_workspace\\future_data\\fu2509_1M.csv',
        'C:\\python_workspace\\future_data\\fu2601_1M.csv'
    ]
    
    # 检查数据文件是否存在
    for path in data_paths:
        if not os.path.exists(path):
            print(f"错误: 找不到数据文件: {path}")
            return
    
    # 创建保存目录
    models_dir = './models_multi_file_optimized'
    plots_dir = './plots_multi_file_optimized'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 实验配置 - 优化的4层LSTM模型
    experiments = {
        '4_layers_128hd_0.2do_optimized': {
            'sequence_length': 20,  # 增加序列长度
            'hidden_dim': 128,      # 增加隐藏层维度
            'num_layers': 4,
            'dropout': 0.2,         # 减少dropout以增加模型容量
            'batch_size': 64,       # 减小批次大小以增加梯度更新频率
            'learning_rate': 0.0005, # 降低学习率
            'epochs': 100,          # 增加训练轮数
            'patience': 20,         # 增加早停耐心值
            'class_weights': [2.0, 0.5, 2.0], # 增加交易信号类别的权重
            'enhanced_features': True, # 使用增强特征
            'dynamic_threshold': True, # 使用动态阈值
            'threshold_percentile': 0.75 # 动态阈值百分位
        },
        '4_layers_256hd_0.2do_optimized': {
            'sequence_length': 20,
            'hidden_dim': 256,      # 更大的隐藏层维度
            'num_layers': 4,
            'dropout': 0.2,
            'batch_size': 64,
            'learning_rate': 0.0003, # 更低的学习率
            'epochs': 100,
            'patience': 20,
            'class_weights': [2.0, 0.5, 2.0],
            'enhanced_features': True,
            'dynamic_threshold': True,
            'threshold_percentile': 0.75
        }
    }
    
    # 加载和预处理数据
    print("\n===== 数据加载和预处理 =====")
    # 使用第一个实验的配置
    first_config = list(experiments.values())[0]
    X_sequences, y_sequences, scaler, feature_names = load_and_preprocess_multi_files(
        data_paths, 
        sequence_length=first_config['sequence_length'],
        config=first_config  # 传递完整配置以便选择特征工程和目标计算方法
    )
    
    input_dim = X_sequences.shape[2]
    print(f"输入维度: {input_dim}")
    print(f"序列总数: {len(X_sequences)}")
    print(f"使用的特征列: {feature_names}")
    print(f"特征数量: {len(feature_names)}")
    
    # 划分训练集、验证集和测试集
    print("\n===== 数据集划分 =====")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 保存预处理参数
    joblib.dump(scaler, os.path.join(models_dir, 'multi_file_scaler.pkl'))
    
    # 存储所有实验结果
    all_results = {}
    
    # 运行每个实验
    for exp_name, config in experiments.items():
        print(f"\n===== 开始实验: {exp_name} =====")
        print(f"配置: {config}")
        
        # 训练模型
        print(f"\n训练模型: {exp_name}")
        model, history = train_model(
            X_train, y_train, X_val, y_val, 
            input_dim, config, exp_name
        )
        
        # 可视化训练历史
        history_plot_path = os.path.join(plots_dir, f'{exp_name}_training_history.png')
        plot_training_history(history, exp_name, history_plot_path)
        
        # 评估模型
        print(f"\n评估模型: {exp_name}")
        evaluation_results = evaluate_model(model, X_test, y_test)
        
        # 打印评估结果
        print(f"\n===== 评估结果: {exp_name} =====")
        print(f"整体准确率: {evaluation_results['overall_accuracy']:.4f}")
        print(f"交易信号准确率: {evaluation_results['signal_accuracy']:.4f}")
        print(f"多单准确率: {evaluation_results['long_accuracy']:.4f}")
        print(f"空单准确率: {evaluation_results['short_accuracy']:.4f}")
        print("\n分类报告:")
        print(evaluation_results['classification_report'])
        print("\n混淆矩阵:")
        print(evaluation_results['confusion_matrix'])
        
        # 可视化混淆矩阵
        cm_plot_path = os.path.join(plots_dir, f'{exp_name}_confusion_matrix.png')
        plot_confusion_matrix(evaluation_results['confusion_matrix'], exp_name, cm_plot_path)
        
        # 保存模型
        model_path = os.path.join(models_dir, f'{exp_name}_model.pkl')
        torch.save(model.state_dict(), model_path)
        print(f"模型保存至: {model_path}")
        
        # 保存完整模型
        full_model_path = os.path.join(models_dir, f'{exp_name}_full_model.pkl')
        torch.save(model, full_model_path)
        print(f"完整模型保存至: {full_model_path}")
        
        # 保存实验结果
        all_results[exp_name] = {
            'config': config,
            'history': history,
            'evaluation': evaluation_results
        }
    
    # 比较所有实验结果
    print("\n===== 实验结果比较 =====")
    comparison_plot_path = os.path.join(plots_dir, 'multi_file_experiment_comparison.png')
    plot_experiment_comparison(all_results, comparison_plot_path)
    
    # 输出最佳模型
    best_signal_accuracy = -1
    best_experiment = None
    
    for exp_name, results in all_results.items():
        signal_acc = results['evaluation']['signal_accuracy']
        if signal_acc > best_signal_accuracy:
            best_signal_accuracy = signal_acc
            best_experiment = exp_name
    
    print(f"\n最佳实验: {best_experiment}")
    print(f"最佳交易信号准确率: {best_signal_accuracy:.4f}")
    print(f"最佳实验配置: {all_results[best_experiment]['config']}")
    
    # 保存所有实验结果
    results_path = os.path.join(models_dir, 'multi_file_experiment_results.pkl')
    joblib.dump(all_results, results_path)
    print(f"所有实验结果保存至: {results_path}")
    
    print("\n===== 实验完成 =====")

if __name__ == "__main__":
    main()