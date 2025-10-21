import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# 1. 加载预处理后的特征数据
def load_engineered_features(features_path='./features/engineered_features.pkl'):
    """加载预处理后的特征数据"""
    data = joblib.load(features_path)
    X = data['X_engineered'].values
    y = data['y'].values
    return X, y

# 2. 数据预处理（适配LSTM输入）
def preprocess_for_lstm(X, y):
    """将数据转换为LSTM需要的形状 (samples, timesteps, features)"""
    # 假设每个样本是一个时间步，特征维度为1（可根据实际需求调整）
    X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    # 将目标变量中的 -1 映射为 2
    y_mapped = np.where(y == -1, 2, y)
    return X_reshaped, y_mapped

# 3. 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# 4. 训练模型
def train_lstm_model(X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """训练LSTM模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = LSTMModel(input_size=X_train.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
            y_pred = (test_outputs > 0.5).float().cpu().numpy()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # 评估模型
    print(classification_report(y_test, y_pred))
    
    # 可视化训练过程
    plt.plot(range(epochs), [loss.item() for loss in train_losses], label='Training Loss')
    plt.plot(range(epochs), [loss.item() for loss in test_losses], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./features/lstm_training_metrics.png')
    
    return model

# 4. 训练模型
def train_lstm_model(X_train, y_train, X_test, y_test):
    """训练LSTM模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = LSTMModel(input_size=X_train.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(20):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
            y_pred = (test_outputs > 0.5).float().cpu().numpy()
            print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # 评估模型
    print(classification_report(y_test, y_pred))
    
    # 可视化训练过程
    plt.plot(range(20), [loss.item() for loss in train_losses], label='Training Loss')
    plt.plot(range(20), [loss.item() for loss in test_losses], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./features/lstm_training_metrics.png')
    
    return model

# 主函数
def main():
    """主函数"""
    # 加载数据
    X, y = load_engineered_features()
    
    # 数据预处理
    X_reshaped, y = preprocess_for_lstm(X, y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = train_lstm_model(X_train, y_train, X_test, y_test)
    
    # 确保目录存在
    os.makedirs('./models/LSTM', exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), './models/LSTM/lstm_model.pth')
    print("LSTM模型训练完成并保存。")

if __name__ == "__main__":
    main()