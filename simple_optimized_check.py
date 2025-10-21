import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

print("===== 简化版优化交易模型性能检查 =====")
print()

# 尝试加载数据
possible_data_paths = [
    './data/processed_data.csv',
    './data/raw_data.csv',
    'data/processed_data.csv',
    'data/raw_data.csv'
]

data_path = None
for path in possible_data_paths:
    if os.path.exists(path):
        data_path = path
        break
    
if data_path is not None and os.path.exists(data_path):
    try:
        data = pd.read_csv(data_path)
        print(f"✓ 成功加载数据: {data_path}")
        print(f"  数据形状: {data.shape}")
        print(f"  可用列: {', '.join(data.columns[:10])}...")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        data = None
else:
    print(f"✗ 找不到数据文件")
    data = None

# 加载优化后的模型
model_path = './models/optimized_trading_model.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"✓ 成功加载优化模型: {model_path}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        model = None
else:
    print(f"✗ 找不到优化模型文件")
    model = None

# 如果数据和模型都成功加载，进行性能评估
if data is not None and model is not None:
    # 确定目标变量
    if 'signal' in data.columns:
        target_col = 'signal'
    elif 'target' in data.columns:
        target_col = 'target'
    else:
        print("✗ 找不到目标变量列'signal'或'target'")
        exit(1)
    
    # 过滤出交易信号样本（类别1和-1）
    trading_data = data[data[target_col] != 0].copy()
    print(f"\n✓ 过滤出交易信号样本: {len(trading_data)} 条")
    print(f"  多单(1)数量: {(trading_data[target_col] == 1).sum()}")
    print(f"  空单(-1)数量: {(trading_data[target_col] == -1).sum()}")
    
    # 准备特征列
    feature_cols = [col for col in data.columns if col != target_col and col != 'date' and col != 'time']
    print(f"✓ 特征列数量: {len(feature_cols)}")
    
    # 分割特征和目标变量
    X = trading_data[feature_cols]
    y_true = trading_data[target_col]
    
    # 预测
    try:
        y_pred = model.predict(X)
        print("✓ 预测完成")
        
        # 计算总体准确率
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f"\n===== 交易信号总体准确率 =====")
        print(f"准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # 详细分类报告
        print("\n===== 交易信号详细分类报告 =====")
        report = classification_report(y_true, y_pred, target_names=['空单(-1)', '多单(1)'])
        print(report)
        
        # 计算多单和空单的准确率
        cm = confusion_matrix(y_true, y_pred)
        
        # 确保混淆矩阵形状正确
        if cm.shape == (2, 2):
            # 假设类别按顺序为[-1, 1]
            short_accuracy = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
            long_accuracy = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
            
            print(f"\n===== 交易信号类别准确率 =====")
            print(f"多单(1)准确率: {long_accuracy:.4f} ({long_accuracy*100:.2f}%)")
            print(f"空单(-1)准确率: {short_accuracy:.4f} ({short_accuracy*100:.2f}%)")
            
            # 评估性能是否达标
            print("\n===== 性能评估 =====")
            if long_accuracy >= 0.5:
                print(f"✓ 多单信号准确率达到 {long_accuracy*100:.2f}%，满足最低要求")
            else:
                print(f"✗ 多单信号准确率仅为 {long_accuracy*100:.2f}%，未达到50%")
                
            if short_accuracy >= 0.5:
                print(f"✓ 空单信号准确率达到 {short_accuracy*100:.2f}%，满足最低要求")
            else:
                print(f"✗ 空单信号准确率仅为 {short_accuracy*100:.2f}%，未达到50%")
                
            # 与原始模型进行对比（如果存在）
            orig_model_path = './models/trained_model.pkl'
            if os.path.exists(orig_model_path):
                try:
                    orig_model = joblib.load(orig_model_path)
                    y_pred_orig = orig_model.predict(X)
                    
                    # 计算原始模型的多单和空单准确率
                    orig_cm = confusion_matrix(y_true, y_pred_orig)
                    if orig_cm.shape == (2, 2):
                        orig_short_accuracy = orig_cm[0, 0] / orig_cm[0, :].sum() if orig_cm[0, :].sum() > 0 else 0
                        orig_long_accuracy = orig_cm[1, 1] / orig_cm[1, :].sum() if orig_cm[1, :].sum() > 0 else 0
                        
                        print("\n===== 与原始模型对比 =====")
                        print(f"原始模型多单准确率: {orig_long_accuracy:.4f} ({orig_long_accuracy*100:.2f}%)")
                        print(f"优化模型多单准确率: {long_accuracy:.4f} ({long_accuracy*100:.2f}%)")
                        print(f"准确率提升: {(long_accuracy - orig_long_accuracy)*100:.2f} 个百分点")
                        
                        print(f"\n原始模型空单准确率: {orig_short_accuracy:.4f} ({orig_short_accuracy*100:.2f}%)")
                        print(f"优化模型空单准确率: {short_accuracy:.4f} ({short_accuracy*100:.2f}%)")
                        print(f"准确率提升: {(short_accuracy - orig_short_accuracy)*100:.2f} 个百分点")
                except Exception as e:
                    print(f"✗ 原始模型对比失败: {e}")
        else:
            print(f"✗ 混淆矩阵形状异常: {cm.shape}")
            print("  原始混淆矩阵:")
            print(cm)
            
    except Exception as e:
        print(f"✗ 预测过程出错: {e}")

print("\n检查完成！")