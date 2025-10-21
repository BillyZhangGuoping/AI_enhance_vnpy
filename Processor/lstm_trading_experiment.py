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
    
    # 改进的学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=config['patience']//2,  # 更积极的学习率调整
        min_lr=1e-6  # 设置最小学习率
    )
    
    # 训练循环
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
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
            current_lr = optimizer.param_groups[0]['lr']
            print(f'轮次 [{epoch+1}/{config["epochs"]}], ' \
                  f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, ' \
                  f'训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}, ' \
                  f'学习率: {current_lr:.6f}')
    
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

# 主函数 - 实验比较
def main():
    print("===== LSTM交易信号预测模型 - 不同层数实验比较 ======")
    
    # 数据路径
    data_path = './processed_data/fu2601_1M_processed_data.csv'
    if not os.path.exists(data_path):
        data_path = './data/fu2601_1M_processed_data.parquet'
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件: {data_path}")
        return
    
    # 创建保存目录
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # 实验配置 - 比较不同层数
    experiments = {
        '2_layers_64hd_0.3do': {
            'sequence_length': 10,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15,
            'class_weights': [1.0, 0.5, 1.0]
        },
        '3_layers_128hd_0.4do': {
            'sequence_length': 10,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.4,
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15,
            'class_weights': [1.0, 0.5, 1.0]
        },
        '4_layers_128hd_0.5do': {
            'sequence_length': 10,
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.5,
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15,
            'class_weights': [1.0, 0.5, 1.0]
        },
        '3_layers_256hd_0.4do': {
            'sequence_length': 10,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.4,
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15,
            'class_weights': [1.0, 0.5, 1.0]
        }
    }
    
    # 加载和预处理数据
    print("\n===== 数据加载和预处理 =====")
    # 使用第一个实验的序列长度
    first_config = list(experiments.values())[0]
    X_sequences, y_sequences, scaler, feature_names = load_and_preprocess_data(
        data_path, 
        sequence_length=first_config['sequence_length']
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
    
    # 保存预处理参数
    joblib.dump(scaler, './models/experiment_scaler.pkl')
    
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
        history_plot_path = f'./plots/{exp_name}_training_history.png'
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
        cm_plot_path = f'./plots/{exp_name}_confusion_matrix.png'
        plot_confusion_matrix(evaluation_results['confusion_matrix'], exp_name, cm_plot_path)
        
        # 保存模型
        model_path = f'./models/{exp_name}_model.pkl'
        torch.save(model.state_dict(), model_path)
        print(f"模型保存至: {model_path}")
        
        # 保存实验结果
        all_results[exp_name] = {
            'config': config,
            'history': history,
            'evaluation': evaluation_results
        }
    
    # 比较所有实验结果
    print("\n===== 实验结果比较 =====")
    comparison_plot_path = './plots/experiment_comparison.png'
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
    results_path = './models/experiment_results.pkl'
    joblib.dump(all_results, results_path)
    print(f"所有实验结果保存至: {results_path}")
    
    print("\n===== 实验完成 =====")

if __name__ == "__main__":
    main()