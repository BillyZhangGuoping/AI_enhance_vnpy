import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report

print("===== 优化交易模型性能评估 =====")
print()

# 查找可用的数据文件
data_files = [
    './processed_data/fu2601_1M_processed_data.csv',
    './processed_data/fu2601_1m_processed_data.parquet',
    './data/fu2601_1M_processed_data.parquet',
    './future_data/fu2601_1M_processed_data.parquet'
]

data = None
data_path = None
for path in data_files:
    if os.path.exists(path):
        try:
            if path.endswith('.parquet'):
                data = pd.read_parquet(path)
            else:
                data = pd.read_csv(path)
            data_path = path
            print(f"✓ 成功加载数据: {path}")
            print(f"  数据形状: {data.shape}")
            break
        except Exception as e:
            print(f"✗ 尝试加载 {path} 失败: {e}")

if data is None:
    print("\n✗ 无法加载任何数据文件，将使用合成数据进行演示评估")
    # 创建合成数据用于演示
    np.random.seed(42)
    n_samples = 1000
    # 假设特征数量为20
    n_features = 20
    X_synthetic = np.random.randn(n_samples, n_features)
    # 创建不平衡的目标变量
    y_synthetic = np.zeros(n_samples)
    y_synthetic[:300] = 1  # 多单
    y_synthetic[300:600] = -1  # 空单
    # 打乱数据
    indices = np.random.permutation(n_samples)
    y_synthetic = y_synthetic[indices]
    
    # 创建合成DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X_synthetic, columns=feature_cols)
    data['signal'] = y_synthetic
    print(f"✓ 创建合成数据: {data.shape}")

# 加载优化模型
model_path = './models/optimized_trading_model.pkl'
if os.path.exists(model_path):
    try:
        model_obj = joblib.load(model_path)
        print(f"\n✓ 成功加载模型文件: {model_path}")
        
        # 检查是否是字典格式并提取模型
        if isinstance(model_obj, dict):
            # 尝试不同的可能键名
            possible_keys = ['model', 'classifier', 'estimator', 'rf_model', 'trading_model']
            model = None
            for key in possible_keys:
                if key in model_obj:
                    model = model_obj[key]
                    print(f"  从字典中提取模型: {key}")
                    break
            
            if model is None:
                print(f"  警告: 模型字典包含键: {list(model_obj.keys())}")
                # 尝试使用第一个值作为模型
                first_key = list(model_obj.keys())[0]
                model = model_obj[first_key]
                print(f"  尝试使用第一个键 '{first_key}' 作为模型")
        else:
            model = model_obj
            print("  模型已直接加载")
            
    except Exception as e:
        print(f"\n✗ 模型加载失败: {e}")
        exit(1)
else:
    print(f"\n✗ 找不到优化模型文件: {model_path}")
    exit(1)

# 准备特征和目标变量
if 'signal' in data.columns:
    target_col = 'signal'
elif 'target' in data.columns:
    target_col = 'target'
else:
    print("\n✗ 找不到目标变量列'signal'或'target'")
    # 假设最后一列是目标变量
    target_col = data.columns[-1]
    print(f"  尝试使用最后一列作为目标变量: {target_col}")

# 过滤出交易信号样本（类别1和-1）
trading_data = data[data[target_col] != 0].copy()
print(f"\n✓ 过滤出交易信号样本: {len(trading_data)} 条")
print(f"  多单(1)数量: {(trading_data[target_col] == 1).sum()}")
print(f"  空单(-1)数量: {(trading_data[target_col] == -1).sum()}")

# 准备特征列（排除目标变量和可能的时间列）
feature_cols = [col for col in data.columns 
                if col != target_col 
                and 'date' not in col.lower() 
                and 'time' not in col.lower()]
print(f"✓ 特征列数量: {len(feature_cols)}")

# 分割特征和目标变量
X = trading_data[feature_cols]
y_true = trading_data[target_col]

# 预测
try:
    y_pred = model.predict(X)
    print("✓ 预测完成")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n===== 混淆矩阵 =====")
    print(f"原始混淆矩阵形状: {cm.shape}")
    
    # 确保我们只关注多单和空单
    if set(y_true.unique()) == {-1, 1}:
        # 重新排列混淆矩阵以确保顺序正确
        classes = sorted(y_true.unique())
        print(f"检测到的类别: {classes}")
        
        # 计算各类别的准确率
        if cm.shape == (2, 2):
            # 假设类别顺序为[-1, 1]或[1, -1]，需要确定正确的索引
            if -1 in classes and 1 in classes:
                if classes[0] == -1 and classes[1] == 1:
                    short_idx = 0
                    long_idx = 1
                else:
                    short_idx = 1
                    long_idx = 0
                
                short_accuracy = cm[short_idx, short_idx] / cm[short_idx, :].sum() if cm[short_idx, :].sum() > 0 else 0
                long_accuracy = cm[long_idx, long_idx] / cm[long_idx, :].sum() if cm[long_idx, :].sum() > 0 else 0
                
                print(f"\n===== 交易信号类别准确率 =====")
                print(f"多单(1)准确率: {long_accuracy:.4f} ({long_accuracy*100:.2f}%)")
                print(f"空单(-1)准确率: {short_accuracy:.4f} ({short_accuracy*100:.2f}%)")
                
                # 详细分类报告
                print("\n===== 交易信号详细分类报告 =====")
                target_names = ['空单(-1)', '多单(1)'] if classes[0] == -1 else ['多单(1)', '空单(-1)']
                report = classification_report(y_true, y_pred, target_names=target_names)
                print(report)
                
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
        else:
            print(f"✗ 混淆矩阵形状不是(2,2)，无法计算准确的类别准确率")
            print("原始混淆矩阵:")
            print(cm)
    else:
        print(f"✗ 检测到的类别不是仅包含-1和1: {set(y_true.unique())}")
        
        # 计算整体准确率
        from sklearn.metrics import accuracy_score
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f"\n整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # 如果有1和-1类别，尝试单独计算它们的准确率
        if 1 in y_true.unique():
            y_true_1 = y_true[y_true == 1]
            y_pred_1 = y_pred[y_true == 1]
            long_accuracy = accuracy_score(y_true_1, y_pred_1)
            print(f"多单(1)准确率: {long_accuracy:.4f} ({long_accuracy*100:.2f}%)")
        
        if -1 in y_true.unique():
            y_true_neg1 = y_true[y_true == -1]
            y_pred_neg1 = y_pred[y_true == -1]
            short_accuracy = accuracy_score(y_true_neg1, y_pred_neg1)
            print(f"空单(-1)准确率: {short_accuracy:.4f} ({short_accuracy*100:.2f}%)")
            
            
    # 与原始模型对比（如果存在）
    orig_model_path = './models/trained_model.pkl'
    if os.path.exists(orig_model_path):
        try:
            orig_model = joblib.load(orig_model_path)
            y_pred_orig = orig_model.predict(X)
            
            print("\n===== 与原始模型对比 =====")
            
            # 计算原始模型的准确率
            if 1 in y_true.unique():
                orig_long_accuracy = accuracy_score(y_true[y_true == 1], y_pred_orig[y_true == 1])
                print(f"原始模型多单准确率: {orig_long_accuracy:.4f} ({orig_long_accuracy*100:.2f}%)")
                if 'long_accuracy' in locals():
                    print(f"优化模型多单准确率: {long_accuracy:.4f} ({long_accuracy*100:.2f}%)")
                    print(f"准确率提升: {(long_accuracy - orig_long_accuracy)*100:.2f} 个百分点")
            
            if -1 in y_true.unique():
                orig_short_accuracy = accuracy_score(y_true[y_true == -1], y_pred_orig[y_true == -1])
                print(f"\n原始模型空单准确率: {orig_short_accuracy:.4f} ({orig_short_accuracy*100:.2f}%)")
                if 'short_accuracy' in locals():
                    print(f"优化模型空单准确率: {short_accuracy:.4f} ({short_accuracy*100:.2f}%)")
                    print(f"准确率提升: {(short_accuracy - orig_short_accuracy)*100:.2f} 个百分点")
                    
        except Exception as e:
            print(f"✗ 原始模型对比失败: {e}")
            
    # 检查是否有可视化文件
    print("\n===== 生成的分析文件 =====")
    if os.path.exists('./models/trading_performance.png'):
        print(f"✓ 交易性能可视化: ./models/trading_performance.png")
        
    # 总结
    print("\n===== 总结 =====")
    if 'long_accuracy' in locals() and 'short_accuracy' in locals():
        if long_accuracy >= 0.5 and short_accuracy >= 0.5:
            print("✓ 优化成功！多单和空单信号的准确率均达到或超过50%")
        elif long_accuracy >= 0.5:
            print("⚠️  部分成功：多单信号准确率达标，但空单信号仍需优化")
        elif short_accuracy >= 0.5:
            print("⚠️  部分成功：空单信号准确率达标，但多单信号仍需优化")
        else:
            print("⚠️  优化未达标：多单和空单信号的准确率均低于50%，需要进一步改进")
            print("  建议：尝试调整特征工程、模型参数或采样策略")
except Exception as e:
    print(f"✗ 预测过程出错: {e}")

print("\n评估完成！")