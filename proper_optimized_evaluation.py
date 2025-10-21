import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

print("===== 优化交易模型性能评估 =====")
print()

# 加载模型和元数据
model_path = './models/optimized_trading_model.pkl'
if os.path.exists(model_path):
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        trained_feature_names = model_data['feature_names']
        print(f"✓ 成功加载模型和元数据")
        print(f"  训练时使用的特征数量: {len(trained_feature_names)}")
        print(f"  前5个训练特征: {trained_feature_names[:5]}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        exit(1)
else:
    print(f"✗ 找不到模型文件: {model_path}")
    exit(1)

# 加载数据
data_path = './processed_data/fu2601_1M_processed_data.csv'
if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
        print(f"\n✓ 成功加载数据: {data_path}")
        print(f"  数据形状: {df.shape}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        exit(1)
else:
    print(f"✗ 找不到数据文件: {data_path}")
    exit(1)

# 复制原始优化模型中的特征工程流程
def prepare_evaluation_data(df):
    # 保存目标变量
    y = df['target'].copy() if 'target' in df.columns else df['signal'].copy()
    
    # 移除不需要的特征（与训练时一致）
    columns_to_drop = [
        'target', 'symbol', 'bob', 'future_close', 'Unnamed: 0'
    ]
    
    available_columns = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=available_columns)
    
    return X, y

def enhanced_feature_engineering(X, lags=[1, 3, 5, 10]):
    X_enhanced = X.copy()
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建滞后特征（与训练时一致）
    for lag in lags:
        for col in numeric_columns:
            X_enhanced[f'{col}_lag_{lag}'] = X[col].shift(lag)
            X_enhanced[f'{col}_pct_change_{lag}'] = X[col].pct_change(periods=lag)
    
    # 创建交叉特征 - 技术指标组合
    if 'rsi' in X_enhanced.columns and 'macd' in X_enhanced.columns:
        X_enhanced['rsi_macd_combined'] = X_enhanced['rsi'] * X_enhanced['macd']
    
    if 'bb_upper' in X_enhanced.columns and 'bb_lower' in X_enhanced.columns:
        X_enhanced['bb_width'] = X_enhanced['bb_upper'] - X_enhanced['bb_lower']
        X_enhanced['bb_width_pct'] = (X_enhanced['bb_width'] / X_enhanced['close']) * 100
    
    # 创建波动率特征组合
    if 'atr' in X_enhanced.columns:
        for window in [5, 10, 20]:
            X_enhanced[f'atr_{window}ma'] = X_enhanced['atr'].rolling(window=window).mean()
    
    # 删除NaN值
    X_enhanced = X_enhanced.dropna()
    
    print(f"增强后特征维度: {X_enhanced.shape}")
    return X_enhanced

def standardize_features(X_enhanced):
    scaler = StandardScaler()
    numeric_columns = X_enhanced.select_dtypes(include=[np.number]).columns
    X_numeric = X_enhanced[numeric_columns].copy()
    
    # 处理异常值
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    # 标准化
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns, index=X_enhanced.index)
    
    return X_scaled_df

# 准备评估数据
print("\n===== 准备评估数据 =====")
X_original, y_original = prepare_evaluation_data(df)
print(f"原始特征维度: {X_original.shape}")

# 应用相同的特征工程
print("\n===== 应用特征工程 =====")
X_enhanced = enhanced_feature_engineering(X_original)

# 调整y的长度以匹配增强后的X
original_len = len(X_original)
enhanced_len = len(X_enhanced)
offset = original_len - enhanced_len
y_adjusted = y_original.iloc[offset:].reset_index(drop=True)
X_enhanced = X_enhanced.reset_index(drop=True)

# 特征标准化
print("\n===== 特征标准化 =====")
X_scaled = standardize_features(X_enhanced)

# 选择与训练时相同的特征
print("\n===== 选择特征子集 =====")
# 找出在评估数据中可用的训练特征
available_features = [f for f in trained_feature_names if f in X_scaled.columns]
missing_features = [f for f in trained_feature_names if f not in X_scaled.columns]

print(f"可用的训练特征: {len(available_features)}/{len(trained_feature_names)}")
if missing_features:
    print(f"缺失的特征 ({len(missing_features)}个): {missing_features[:5]}..." if len(missing_features) > 5 else missing_features)

# 如果可用特征太少，无法进行有意义的评估
if len(available_features) < len(trained_feature_names) * 0.5:
    print("\n✗ 警告: 太多训练特征在评估数据中不可用，可能导致评估结果不准确")

# 使用可用的训练特征
X_final = X_scaled[available_features].copy()
print(f"评估时使用的特征数量: {X_final.shape[1]}")

# 过滤出交易信号样本（类别1和-1）
print("\n===== 过滤交易信号样本 =====")
signal_mask = (y_adjusted == 1) | (y_adjusted == -1)
X_signal = X_final[signal_mask]
y_signal = y_adjusted[signal_mask]

print(f"交易信号样本总数: {len(y_signal)}")
print(f"多单(1)数量: {(y_signal == 1).sum()}")
print(f"空单(-1)数量: {(y_signal == -1).sum()}")

# 预测
try:
    print("\n===== 执行预测 =====")
    y_pred = model.predict(X_signal)
    print("✓ 预测完成")
    
    # 计算总体准确率
    overall_accuracy = accuracy_score(y_signal, y_pred)
    print(f"\n===== 交易信号总体准确率 =====")
    print(f"准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # 过滤掉可能的0类别预测，确保只考虑-1和1
    valid_classes_mask = (y_pred == 1) | (y_pred == -1)
    y_pred_filtered = y_pred[valid_classes_mask]
    y_signal_filtered = y_signal[valid_classes_mask]
    
    print(f"过滤后评估样本数: {len(y_pred_filtered)}")
    print(f"过滤掉的预测类别: {len(y_pred) - len(y_pred_filtered)}")
    
    # 详细分类报告
    print("\n===== 交易信号详细分类报告 =====")
    report = classification_report(y_signal_filtered, y_pred_filtered, labels=[-1, 1], target_names=['空单(-1)', '多单(1)'])
    print(report)
    
    # 混淆矩阵 - 已在过滤后提供
    
    # 计算多单和空单的准确率
    cm_filtered = confusion_matrix(y_signal_filtered, y_pred_filtered, labels=[-1, 1])
    long_accuracy = cm_filtered[1,1] / cm_filtered[1,:].sum() if cm_filtered[1,:].sum() > 0 else 0
    short_accuracy = cm_filtered[0,0] / cm_filtered[0,:].sum() if cm_filtered[0,:].sum() > 0 else 0
    
    print("\n===== 过滤后混淆矩阵 =====")
    cm_filtered_df = pd.DataFrame(cm_filtered, index=['实际空单(-1)', '实际多单(1)'], columns=['预测空单(-1)', '预测多单(1)'])
    print(cm_filtered_df)
    
    print(f"\n===== 交易信号类别准确率 =====")
    print(f"多单(1)准确率: {long_accuracy:.4f} ({long_accuracy*100:.2f}%)")
    print(f"空单(-1)准确率: {short_accuracy:.4f} ({short_accuracy*100:.2f}%)")
    
    # 与原始模型对比（如果存在）
    orig_model_path = './models/trained_model.pkl'
    if os.path.exists(orig_model_path):
        try:
            orig_model_obj = joblib.load(orig_model_path)
            # 处理原始模型可能也是字典格式
            if isinstance(orig_model_obj, dict):
                if 'model' in orig_model_obj:
                    orig_model = orig_model_obj['model']
                else:
                    # 尝试使用第一个键
                    orig_model = orig_model_obj[list(orig_model_obj.keys())[0]]
            else:
                orig_model = orig_model_obj
                
            y_pred_orig = orig_model.predict(X_signal)
            
            # 过滤原始模型的预测结果
            orig_valid_classes_mask = (y_pred_orig == 1) | (y_pred_orig == -1)
            y_pred_orig_filtered = y_pred_orig[orig_valid_classes_mask]
            y_signal_orig_filtered = y_signal[orig_valid_classes_mask]
            
            orig_long_accuracy = sum((y_signal_orig_filtered == 1) & (y_pred_orig_filtered == 1)) / sum(y_signal_orig_filtered == 1) if sum(y_signal_orig_filtered == 1) > 0 else 0
            orig_short_accuracy = sum((y_signal_orig_filtered == -1) & (y_pred_orig_filtered == -1)) / sum(y_signal_orig_filtered == -1) if sum(y_signal_orig_filtered == -1) > 0 else 0
            
            print("\n===== 与原始模型对比 =====")
            print(f"原始模型多单准确率: {orig_long_accuracy:.4f} ({orig_long_accuracy*100:.2f}%)")
            print(f"优化模型多单准确率: {long_accuracy:.4f} ({long_accuracy*100:.2f}%)")
            print(f"准确率提升: {(long_accuracy - orig_long_accuracy)*100:.2f} 个百分点")
            
            print(f"\n原始模型空单准确率: {orig_short_accuracy:.4f} ({orig_short_accuracy*100:.2f}%)")
            print(f"优化模型空单准确率: {short_accuracy:.4f} ({short_accuracy*100:.2f}%)")
            print(f"准确率提升: {(short_accuracy - orig_short_accuracy)*100:.2f} 个百分点")
            
        except Exception as e:
            print(f"\n✗ 原始模型对比失败: {e}")
    
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
        
    # 总结
    print("\n===== 总结 =====")
    if long_accuracy >= 0.5 and short_accuracy >= 0.5:
        print("✓ 优化成功！多单和空单信号的准确率均达到或超过50%")
    elif long_accuracy >= 0.5:
        print("⚠️  部分成功：多单信号准确率达标，但空单信号仍需优化")
    elif short_accuracy >= 0.5:
        print("⚠️  部分成功：空单信号准确率达标，但多单信号仍需优化")
    else:
        print("⚠️  优化未达标：多单和空单信号的准确率均低于50%，需要进一步改进")
        
    # 检查是否有可视化文件
    print("\n===== 分析文件 =====")
    if os.path.exists('./models/trading_performance.png'):
        print(f"✓ 交易性能可视化: ./models/trading_performance.png")
        
    # 如果有模型元数据中的交叉验证结果，显示它们
    if 'fold_details' in model_data:
        print("\n===== 训练时交叉验证结果 =====")
        fold_details = pd.DataFrame(model_data['fold_details'])
        avg_precision_1 = fold_details['precision_1'].mean()
        avg_precision_neg1 = fold_details['precision_neg1'].mean()
        
        print(f"平均多单(1)精确率: {avg_precision_1:.4f} ({avg_precision_1*100:.2f}%)")
        print(f"平均空单(-1)精确率: {avg_precision_neg1:.4f} ({avg_precision_neg1*100:.2f}%)")
        
        avg_recall_1 = fold_details['recall_1'].mean()
        avg_recall_neg1 = fold_details['recall_neg1'].mean()
        
        print(f"平均多单(1)召回率: {avg_recall_1:.4f} ({avg_recall_1*100:.2f}%)")
        print(f"平均空单(-1)召回率: {avg_recall_neg1:.4f} ({avg_recall_neg1*100:.2f}%)")
        
        if 'signal_precisions' in model_data:
            avg_signal_precision = np.mean(model_data['signal_precisions'])
            print(f"平均交易信号准确率: {avg_signal_precision:.4f} ({avg_signal_precision*100:.2f}%)")
    
except Exception as e:
    print(f"\n✗ 预测过程出错: {e}")
    import traceback
    traceback.print_exc()

print("\n评估完成！")