import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.utils.class_weight import compute_class_weight

# 1. 加载数据
def load_data():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_script_dir, 'processed_data', 'fu2601_1M_processed_data.parquet')
    
    df = pd.read_parquet(data_path)
    df = df.iloc[50:].reset_index(drop=True)
    return df

# 2. 准备数据 - 专注于交易信号类别
def prepare_trading_data():
    df = load_data()
    
    # 保存目标变量
    y = df['target'].copy()
    
    # 移除不需要的特征
    columns_to_drop = [
        'target', 'symbol', 'bob', 'future_close',
        'Unnamed: 0'
    ]
    
    available_columns = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=available_columns)
    
    print(f"原始特征维度: {X.shape}")
    print(f"目标变量分布:")
    print(y.value_counts())
    print("\n交易信号分布:")
    print(f"开多单(1): {sum(y == 1)} 个样本 ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"开空单(-1): {sum(y == -1)} 个样本 ({sum(y == -1)/len(y)*100:.1f}%)")
    print(f"不开单(0): {sum(y == 0)} 个样本 ({sum(y == 0)/len(y)*100:.1f}%)")
    
    return X, y, df

# 3. 增强特征工程 - 专注于交易信号相关特征
def enhanced_feature_engineering(X, lags=[1, 3, 5, 10]):
    X_enhanced = X.copy()
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建滞后特征
    for lag in lags:
        for col in numeric_columns:
            X_enhanced[f'{col}_lag_{lag}'] = X[col].shift(lag)
            
            # 创建多种变化率特征
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

# 4. 特征选择 - 专注于交易信号预测
def select_trading_features(X, y, n_features=20):
    # 首先过滤掉0类，专注于1和-1类的预测
    signal_mask = (y == 1) | (y == -1)
    X_signal = X[signal_mask]
    y_signal = y[signal_mask]
    
    # 使用随机森林进行特征重要性评估
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_signal, y_signal)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 选择最重要的特征
    selected_features = feature_importance.head(n_features)['feature'].tolist()
    print(f"\n选择的Top {n_features} 特征:")
    for i, (feature, importance) in enumerate(zip(
            feature_importance.head(n_features)['feature'], 
            feature_importance.head(n_features)['importance']
    )):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # 处理多重共线性
    X_top = X[selected_features]
    corr_matrix = X_top.corr().abs()
    
    # 选择要保留的特征（移除高相关特征）
    selected_final = []
    dropped = []
    
    for feature in selected_features:
        if feature not in dropped:
            selected_final.append(feature)
            # 找到与当前特征高度相关的其他特征
            correlated = corr_matrix[corr_matrix[feature] > 0.80].index.tolist()
            for corr_feature in correlated:
                if corr_feature != feature and corr_feature in selected_features:
                    if corr_feature not in dropped:
                        dropped.append(corr_feature)
    
    print(f"\n最终选择特征数量: {len(selected_final)}")
    print(f"因共线性移除的特征: {dropped}")
    
    return X[selected_final], selected_final

# 5. 高级类别平衡
def advanced_class_balancing(X_train, y_train, method='smoteenn'):
    if method == 'smote':
        # SMOTE过采样
        sampler = SMOTE(random_state=42, k_neighbors=5)
    elif method == 'adasyn':
        # ADASYN自适应过采样
        sampler = ADASYN(random_state=42, n_neighbors=5)
    elif method == 'smoteenn':
        # SMOTE+ENN组合采样（过采样+欠采样）
        sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=5))
    else:
        return X_train, y_train
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"类别平衡后样本分布:")
        print(pd.Series(y_resampled).value_counts())
        return X_resampled, y_resampled
    except Exception as e:
        print(f"类别平衡失败: {e}, 使用原始数据")
        return X_train, y_train

# 6. 优化的模型训练 - 专注于交易信号准确率
def train_optimized_trading_model(X, y):
    print("\n=== 开始优化模型训练 ===")
    
    # 计算类别权重，特别强调交易信号类别（1和-1）
    class_weights = {}
    # 提高交易信号类别的权重
    total_samples = len(y)
    class_counts = y.value_counts()
    
    # 为交易信号类别（1和-1）设置更高的权重
    for cls in [-1, 0, 1]:
        if cls == 0:
            # 降低0类的权重
            class_weights[cls] = 0.5
        else:
            # 提高交易信号类别的权重
            class_weights[cls] = total_samples / (class_counts[cls] * 1.5)
    
    print(f"自定义类别权重: {class_weights}")
    
    # 使用时间序列分割
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 优化的随机森林参数
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        max_features='sqrt'
    )
    
    # 交叉验证评估，专注于交易信号准确率
    cv_scores = []
    signal_precisions = []
    signal_recalls = []
    fold_details = []
    
    print("\n=== 交叉验证开始 ===")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 应用类别平衡
        X_train_balanced, y_train_balanced = advanced_class_balancing(X_train, y_train, method='smoteenn')
        
        # 训练模型
        rf.fit(X_train_balanced, y_train_balanced)
        y_pred = rf.predict(X_test)
        
        # 计算常规指标
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores.append(weighted_f1)
        
        # 计算交易信号特定指标
        # 只考虑1和-1类的准确率
        signal_mask_test = (y_test == 1) | (y_test == -1)
        if sum(signal_mask_test) > 0:
            signal_precision = precision_score(y_test[signal_mask_test], y_pred[signal_mask_test], average='macro')
            signal_recall = recall_score(y_test[signal_mask_test], y_pred[signal_mask_test], average='macro')
            signal_precisions.append(signal_precision)
            signal_recalls.append(signal_recall)
        
        # 获取交易信号类别的详细指标
        y_test_signal = y_test[signal_mask_test]
        y_pred_signal = y_pred[signal_mask_test]
        
        # 计算1和-1类的单独准确率
        precision_1 = precision_score(y_test, y_pred, labels=[1], average=None)[0] if 1 in y_test.unique() else 0
        precision_neg1 = precision_score(y_test, y_pred, labels=[-1], average=None)[0] if -1 in y_test.unique() else 0
        
        recall_1 = recall_score(y_test, y_pred, labels=[1], average=None)[0] if 1 in y_test.unique() else 0
        recall_neg1 = recall_score(y_test, y_pred, labels=[-1], average=None)[0] if -1 in y_test.unique() else 0
        
        fold_details.append({
            'fold': fold+1,
            'weighted_f1': weighted_f1,
            'signal_precision': signal_precision if sum(signal_mask_test) > 0 else 0,
            'signal_recall': signal_recall if sum(signal_mask_test) > 0 else 0,
            'precision_1': precision_1,
            'precision_neg1': precision_neg1,
            'recall_1': recall_1,
            'recall_neg1': recall_neg1
        })
        
        print(f"折 {fold+1} - 加权F1: {weighted_f1:.4f}")
        print(f"        交易信号准确率: {signal_precision:.4f}")
        print(f"        多单(1)精确率: {precision_1:.4f}, 召回率: {recall_1:.4f}")
        print(f"        空单(-1)精确率: {precision_neg1:.4f}, 召回率: {recall_neg1:.4f}")
    
    print(f"\n平均加权F1分数: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"平均交易信号准确率: {np.mean(signal_precisions):.4f} (+/- {np.std(signal_precisions):.4f})")
    print(f"平均交易信号召回率: {np.mean(signal_recalls):.4f} (+/- {np.std(signal_recalls):.4f})")
    
    # 最终模型训练
    print("\n=== 最终模型训练 ===")
    X_balanced, y_balanced = advanced_class_balancing(X, y, method='smoteenn')
    rf_final = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        max_features='sqrt'
    )
    
    rf_final.fit(X_balanced, y_balanced)
    y_pred_final = rf_final.predict(X)
    
    # 评估交易信号性能
    signal_mask = (y == 1) | (y == -1)
    y_true_signal = y[signal_mask]
    y_pred_signal = y_pred_final[signal_mask]
    
    print("\n=== 最终模型交易信号性能 ===")
    print(f"交易信号样本数: {len(y_true_signal)}")
    print(f"交易信号准确率: {precision_score(y_true_signal, y_pred_signal, average='macro'):.4f}")
    
    # 单独计算多单和空单的性能
    print("\n多单(1)性能:")
    if 1 in y_true_signal.unique():
        precision_1 = precision_score(y, y_pred_final, labels=[1], average=None)[0]
        recall_1 = recall_score(y, y_pred_final, labels=[1], average=None)[0]
        print(f"精确率: {precision_1:.4f}")
        print(f"召回率: {recall_1:.4f}")
        print(f"F1分数: {2*(precision_1*recall_1)/(precision_1+recall_1) if (precision_1+recall_1) > 0 else 0:.4f}")
    
    print("\n空单(-1)性能:")
    if -1 in y_true_signal.unique():
        precision_neg1 = precision_score(y, y_pred_final, labels=[-1], average=None)[0]
        recall_neg1 = recall_score(y, y_pred_final, labels=[-1], average=None)[0]
        print(f"精确率: {precision_neg1:.4f}")
        print(f"召回率: {recall_neg1:.4f}")
        print(f"F1分数: {2*(precision_neg1*recall_neg1)/(precision_neg1+recall_neg1) if (precision_neg1+recall_neg1) > 0 else 0:.4f}")
    
    # 保存模型和元数据
    model_data = {
        'model': rf_final,
        'feature_names': X.columns.tolist(),
        'cv_scores': cv_scores,
        'signal_precisions': signal_precisions,
        'signal_recalls': signal_recalls,
        'fold_details': fold_details,
        'class_weights': class_weights,
        'training_date': pd.Timestamp.now()
    }
    
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model_data, './models/optimized_trading_model.pkl')
    
    print("\n优化模型训练完成并保存!")
    return model_data

# 7. 可视化交易信号性能
def plot_trading_performance(model_data):
    plt.figure(figsize=(15, 12))
    
    # 1. 交叉验证交易信号准确率
    plt.subplot(2, 2, 1)
    folds = range(1, len(model_data['signal_precisions']) + 1)
    plt.plot(folds, model_data['signal_precisions'], 'bo-', label='交易信号准确率')
    plt.axhline(y=0.5, color='r', linestyle='--', label='50%基准线')
    plt.xlabel('折数')
    plt.ylabel('准确率')
    plt.title('交叉验证交易信号准确率')
    plt.legend()
    plt.grid(True)
    
    # 2. 多单和空单性能
    plt.subplot(2, 2, 2)
    fold_details = pd.DataFrame(model_data['fold_details'])
    plt.plot(folds, fold_details['precision_1'], 'go-', label='多单(1)精确率')
    plt.plot(folds, fold_details['precision_neg1'], 'ro-', label='空单(-1)精确率')
    plt.axhline(y=0.5, color='r', linestyle='--', label='50%基准线')
    plt.xlabel('折数')
    plt.ylabel('精确率')
    plt.title('多单和空单精确率对比')
    plt.legend()
    plt.grid(True)
    
    # 3. 特征重要性
    plt.subplot(2, 2, 3)
    feature_importance = pd.DataFrame({
        'feature': model_data['feature_names'],
        'importance': model_data['model'].feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('重要性')
    plt.title('Top 10交易信号预测特征')
    
    # 4. 类别权重可视化
    plt.subplot(2, 2, 4)
    weights = model_data['class_weights']
    plt.bar(['空单(-1)', '不开单(0)', '多单(1)'], [weights[-1], weights[0], weights[1]])
    plt.xlabel('类别')
    plt.ylabel('权重')
    plt.title('模型类别权重分布')
    
    plt.tight_layout()
    plt.savefig('./models/trading_performance.png', dpi=300, bbox_inches='tight')
    print("交易性能可视化已保存到 ./models/trading_performance.png")

# 主函数
def main():
    print("=== 开始优化交易策略模型训练 ===")
    
    # 1. 准备数据
    X, y, original_df = prepare_trading_data()
    
    # 2. 增强特征工程
    X_enhanced = enhanced_feature_engineering(X)
    
    # 调整y的长度
    y_adjusted = y.iloc[X.shape[0] - X_enhanced.shape[0]:].reset_index(drop=True)
    X_enhanced = X_enhanced.reset_index(drop=True)
    
    # 3. 特征标准化
    print("\n=== 特征标准化 ===")
    scaler = StandardScaler()
    numeric_columns = X_enhanced.select_dtypes(include=[np.number]).columns
    X_numeric = X_enhanced[numeric_columns].copy()
    
    # 处理异常值
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    # 标准化
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns, index=X_enhanced.index)
    
    # 4. 特征选择
    print("\n=== 交易特征选择 ===")
    X_selected, feature_names = select_trading_features(X_scaled_df, y_adjusted, n_features=25)
    
    # 5. 训练优化模型
    model_data = train_optimized_trading_model(X_selected, y_adjusted)
    
    # 6. 可视化性能
    plot_trading_performance(model_data)
    
    print("\n=== 优化交易策略模型训练完成 ===")

if __name__ == "__main__":
    main()