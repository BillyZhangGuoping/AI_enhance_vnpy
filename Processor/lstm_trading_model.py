import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import os
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

# 数据加载和预处理
def load_and_preprocess_data(data_path, sequence_length=10):
    print(f"正在加载数据: {data_path}")
    # 支持多种格式的数据
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")
    
    print(f"数据形状: {df.shape}")
    
    # 准备特征和标签
    if 'target' in df.columns:
        y = df['target'].values
    elif 'signal' in df.columns:
        y = df['signal'].values
    else:
        raise ValueError("数据中没有找到'target'或'signal'列")
    
    # 移除不需要的列
    columns_to_drop = ['target', 'signal', 'symbol', 'bob', 'future_close', 'Unnamed: 0']
    available_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=available_columns_to_drop)
    
    # 只保留数值列
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].copy()
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建序列数据
    X_sequences, y_sequences = create_sequences(X_scaled, y, sequence_length)
    
    return X_sequences, y_sequences, scaler, df.columns.tolist()

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
    # 假设标签是-1, 0, 1
    return {-1: 0, 0: 1, 1: 2}

def get_reverse_label_mapping():
    return {0: -1, 1: 0, 2: 1}

# 训练模型
def train_model(X_train, y_train, X_val, y_val, input_dim, config):
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 初始化模型
    model = LSTMTradingModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        output_dim=3,  # 三个类别: -1, 0, 1
        dropout=config['dropout']
    )
    
    # 使用GPU如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config['class_weights'], device=device))
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 计算准确率
        val_accuracy = accuracy_score(all_targets, all_preds)
        
        # 学习率调度
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
        
        if (epoch + 1) % 10 == 0:
            print(f'轮次 [{epoch+1}/{config["epochs"]}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}')
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    
    return model, train_losses, val_losses

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
def plot_training_history(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('模型训练历史')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史图保存至: {save_path}")
    else:
        plt.show()

# 可视化混淆矩阵
def plot_confusion_matrix(cm, save_path=None):
    # 确保只使用交易信号类别(-1,1)的混淆矩阵部分
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Trading Signal Confusion Matrix')
    plt.colorbar()
    
    classes = ['Short(-1)', 'Long(1)']
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
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

# 主函数
def main():
    print("===== LSTM交易信号预测模型 ======")
    
    # 数据路径
    data_path = './processed_data/fu2601_1M_processed_data.csv'
    if not os.path.exists(data_path):
        data_path = './data/fu2601_1M_processed_data.parquet'
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件: {data_path}")
        return
    
    # 配置参数
    config = {
        'sequence_length': 10,  # 序列长度
        'hidden_dim': 64,       # 隐藏层维度
        'num_layers': 2,        # LSTM层数
        'dropout': 0.3,         # Dropout率
        'batch_size': 64,       # 批次大小
        'learning_rate': 0.001, # 学习率
        'epochs': 100,          # 训练轮次
        'patience': 15,         # 早停耐心值
        'class_weights': [1.0, 0.5, 1.0]  # 类别权重，给交易信号(-1,1)更高权重
    }
    
    # 加载和预处理数据
    print("\n===== 数据加载和预处理 =====")
    X_sequences, y_sequences, scaler, feature_names = load_and_preprocess_data(
        data_path, 
        sequence_length=config['sequence_length']
    )
    
    input_dim = X_sequences.shape[2]
    print(f"输入维度: {input_dim}")
    print(f"序列总数: {len(X_sequences)}")
    
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
    
    # 训练模型
    print("\n===== 模型训练 =====")
    model, train_losses, val_losses = train_model(
        X_train, y_train, X_val, y_val, input_dim, config
    )
    
    # 可视化训练历史
    os.makedirs('./plots', exist_ok=True)
    plot_training_history(
        train_losses, 
        val_losses, 
        save_path='./plots/lstm_training_history.png'
    )
    
    # 评估模型
    print("\n===== 模型评估 =====")
    eval_results = evaluate_model(model, X_test, y_test)
    
    # 打印评估结果
    print(f"\n===== 评估结果 =====")
    print(f"整体准确率: {eval_results['overall_accuracy']:.4f} ({eval_results['overall_accuracy']*100:.2f}%)")
    print(f"交易信号准确率: {eval_results['signal_accuracy']:.4f} ({eval_results['signal_accuracy']*100:.2f}%)")
    print(f"多单准确率: {eval_results['long_accuracy']:.4f} ({eval_results['long_accuracy']*100:.2f}%)")
    print(f"空单准确率: {eval_results['short_accuracy']:.4f} ({eval_results['short_accuracy']*100:.2f}%)")
    
    print("\n===== 分类报告 =====")
    print(eval_results['classification_report'])
    
    print("\n===== 混淆矩阵 =====")
    # 直接打印混淆矩阵的数值，避免形状不匹配的问题
    print(f"{eval_results['confusion_matrix']}")
    print("混淆矩阵格式: [[TP, FP], [FN, TN]]")
    print("其中: 行表示实际类别(-1,1), 列表示预测类别(-1,1)")
    
    # 可视化混淆矩阵
    plot_confusion_matrix(
        eval_results['confusion_matrix'], 
        save_path='./plots/lstm_confusion_matrix.png'
    )
    
    # 保存模型
    print("\n===== 保存模型 =====")
    os.makedirs('./models', exist_ok=True)
    
    # 准备模型数据
    model_data = {
        'model': model,
        'config': config,
        'scaler': scaler,
        'sequence_length': config['sequence_length'],
        'feature_names': feature_names
    }
    
    # 保存完整模型
    joblib.dump(model_data, './models/lstm_trading_model.pkl')
    print("模型已保存至: ./models/lstm_trading_model.pkl")
    
    # 评估性能是否达标
    print("\n===== 性能评估 =====")
    if eval_results['long_accuracy'] >= 0.5:
        print(f"✓ 多单信号准确率达到 {eval_results['long_accuracy']*100:.2f}%，满足最低要求")
    else:
        print(f"✗ 多单信号准确率仅为 {eval_results['long_accuracy']*100:.2f}%，未达到50%")
        
    if eval_results['short_accuracy'] >= 0.5:
        print(f"✓ 空单信号准确率达到 {eval_results['short_accuracy']*100:.2f}%，满足最低要求")
    else:
        print(f"✗ 空单信号准确率仅为 {eval_results['short_accuracy']*100:.2f}%，未达到50%")
    
    # 总结
    print("\n===== 总结 =====")
    if eval_results['long_accuracy'] >= 0.5 and eval_results['short_accuracy'] >= 0.5:
        print("✓ 优化成功！LSTM模型的多单和空单信号准确率均达到或超过50%")
    elif eval_results['long_accuracy'] >= 0.5:
        print("⚠️  部分成功：多单信号准确率达标，但空单信号仍需优化")
    elif eval_results['short_accuracy'] >= 0.5:
        print("⚠️  部分成功：空单信号准确率达标，但多单信号仍需优化")
    else:
        print("⚠️  优化未达标：多单和空单信号的准确率均低于50%，需要进一步改进")

if __name__ == "__main__":
    main()