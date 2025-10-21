import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE  # 用于过采样
import matplotlib.pyplot as plt
import seaborn as sns

# 0. 读取 Parquet 数据
def load_data():
    # 获取当前脚本所在目录的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建指向上一级目录中目标文件的绝对路径
    parent_dir = os.path.dirname(current_script_dir)
    data_path = os.path.join(parent_dir, 'processed_data', 'fu2601_1M_processed_data.parquet')
    
    df = pd.read_parquet(data_path)
    # 删除前50行数据，解决空值问题
    df = df.iloc[50:].reset_index(drop=True)
    return df

# 1. 加载数据并分离target
def load_and_prepare_data(csv_path):
    """加载数据并正确分离特征和目标变量"""
    df = load_data()
    
    # 保存目标变量
    y = df['target'].copy()
    
    # 移除不需要的特征
    columns_to_drop = [
        'target', 'symbol', 'bob', 'future_close',  # 目标变量和标识列
        'Unnamed: 0'  # 可能的索引列
    ]
    
    # 只保留实际可用的列
    available_columns = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=available_columns)
    
    print(f"特征维度: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts()}")
    
    return X, y, df

# 2. 创建滞后特征的意义和实现
def create_lag_features(X, lags=[1, 2, 3,4,5]):
    """
    创建滞后特征的意义：
    - 捕捉时间序列的动量效应：价格变动往往具有连续性
    - 识别趋势变化：通过比较当前值与历史值判断趋势强度
    - 提供模型记忆能力：让模型"看到"过去几个周期的模式
    """
    X_lagged = X.copy()
    
    # 选择数值型特征创建滞后项（排除分类特征如hour, minute等）
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    for lag in lags:
        for col in numeric_columns:
            # 创建滞后特征
            X_lagged[f'{col}_lag_{lag}'] = X[col].shift(lag)
            
            # 创建滞后变化率特征（更有意义）
            if lag == 1:  # 只对lag=1创建变化率，避免多重共线性
                X_lagged[f'{col}_pct_change_{lag}'] = X[col].pct_change(periods=lag)
    
    # 删除因滞后产生的NaN行
    X_lagged = X_lagged.iloc[max(lags):]
    
    print(f"滞后特征创建完成，新维度: {X_lagged.shape}")
    return X_lagged

# 3. 完整的特征工程流程
def feature_engineering_pipeline(csv_path, output_path='./features/engineered_features.pkl'):
    """完整的特征工程流程"""
    
    # 1. 加载数据
    X, y, original_df = load_and_prepare_data(csv_path)
    
    # 2. 创建滞后特征
    X_with_lags = create_lag_features(X)
    
    # 调整y的长度（因为滞后特征删除了前几行）
    y_adjusted = y.iloc[max([1,2,3,4,5]):].reset_index(drop=True)
    
    # 3. 特征标准化
    print("=== 特征标准化 ===")
    scaler = StandardScaler()
    
    # 获取数值型列（排除可能存在的非数值列）
    numeric_columns = X_with_lags.select_dtypes(include=[np.number]).columns
    X_numeric = X_with_lags[numeric_columns]
    
    # 检查并处理无穷大或极大值
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    # 标准化数值特征
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns, index=X_with_lags.index)
    
    # 4. 特征选择 - 随机森林重要性
    print("=== 特征选择 ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled_df, y_adjusted)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': numeric_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 选择top 15特征
    top_15_features = feature_importance.head(15)['feature'].tolist()
    print("Top 15特征:", top_15_features)
    
    # 5. 处理多重共线性
    print("=== 处理多重共线性 ===")
    X_top = X_scaled_df[top_15_features]
    
    # 计算相关系数矩阵
    corr_matrix = X_top.corr().abs()
    
    # 选择要保留的特征（移除高相关特征）
    selected_features = []
    dropped_features = []
    
    for feature in top_15_features:
        if feature not in dropped_features:
            selected_features.append(feature)
            # 找到与当前特征高度相关的其他特征
            correlated_features = corr_matrix[corr_matrix[feature] > 0.85].index.tolist()
            # 从待选列表中移除这些特征（除了自己）
            for corr_feature in correlated_features:
                if corr_feature != feature and corr_feature in top_15_features:
                    if corr_feature not in dropped_features:
                        dropped_features.append(corr_feature)
    
    print(f"最终选择特征数量: {len(selected_features)}")
    print(f"因共线性移除的特征: {dropped_features}")
    
    X_final = X_top[selected_features]
    
    # 6. 保存处理结果
    result = {
        'X_engineered': X_final,
        'y': y_adjusted,
        'feature_names': selected_features,
        'scaler': scaler,
        'original_shape': X.shape
    }
    
    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    joblib.dump(result, output_path)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_10 = feature_importance.head(10)
    plt.barh(top_10['feature'], top_10['importance'])
    plt.xlabel('特征重要性')
    plt.title('Top 10 特征重要性排序')
    plt.tight_layout()
    plt.savefig('./features/feature_importance.png')
    
    # 可视化相关系数矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_final.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('最终特征相关系数矩阵')
    plt.tight_layout()
    plt.savefig('./features/correlation_matrix.png')
    
    print(f"\n=== 特征工程完成 ===")
    print(f"原始特征数: {X.shape[1]}")
    print(f"工程后特征数: {X_final.shape[1]}")
    print(f"数据保存至: {output_path}")
    
    return result

# 4. 模型训练示例（使用处理后的特征）
def train_model_with_engineered_features(features_path):
    """使用处理后的特征进行模型训练 - 优化多分类不平衡版本"""
    
    # 加载工程化特征
    data = joblib.load(features_path)
    X = data['X_engineered']
    y = data['y']
    
    print("=== 开始模型训练 ===")
    print(f"训练数据形状: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts()}")
    
    # 计算类别权重（基于目标分布）
    class_counts = y.value_counts()
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    print(f"计算的类别权重: {class_weights}")
    
    # 使用时间序列分割（避免未来信息泄露）
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 使用更鲁棒的随机森林参数
    rf = RandomForestClassifier(
        n_estimators=200,  # 增加树的数量
        max_depth=10,      # 限制深度防止过拟合
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',  # 自动平衡类别权重
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心
    )
    
    cv_scores = []
    fold_details = []
    
    print("\n=== 交叉验证开始 ===")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 可选：使用SMOTE过采样（针对时间序列需谨慎）
        # 仅在训练集上应用，避免数据泄露
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"Fold {fold+1} SMOTE应用后: {X_train_resampled.shape}")
        except Exception as e:
            print(f"Fold {fold+1} SMOTE失败: {e}, 使用原始数据")
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # 训练模型
        rf.fit(X_train_resampled, y_train_resampled)
        y_pred = rf.predict(X_test)
        
        # 使用加权F1分数（解决多分类问题）
        score = f1_score(y_test, y_pred, average='weighted')  # 关键修改
        cv_scores.append(score)
        
        # 保存每折的详细结果
        fold_details.append({
            'fold': fold+1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'f1_weighted': score,
            'y_test_distribution': y_test.value_counts().to_dict()
        })
        
        print(f"Fold {fold+1} - 加权F1分数: {score:.4f}")
        print(f"          测试集分布: {dict(y_test.value_counts())}")
    
    print(f"\n平均加权F1分数: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # 最终模型训练（使用全部数据）
    print("\n=== 最终模型训练 ===")
    rf_final = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_final.fit(X, y)
    
    # 最终模型评估
    y_pred_final = rf_final.predict(X)
    final_report = classification_report(y, y_pred_final, output_dict=True)
    
    print("\n=== 最终模型性能 ===")
    print(classification_report(y, y_pred_final))
    
    # 特征重要性分析
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10特征重要性:")
    print(feature_importance.head(10))
    
    # 保存模型和元数据
    model_data = {
        'model': rf_final,
        'feature_names': X.columns.tolist(),
        'feature_importance': feature_importance,
        'cv_scores': cv_scores,
        'fold_details': fold_details,
        'classification_report': final_report,
        'class_distribution': y.value_counts().to_dict(),
        'training_date': pd.Timestamp.now()
    }
    
    # 确保目录存在
    import os
    os.makedirs('./models', exist_ok=True)
    
    joblib.dump(model_data, './models/trained_model.pkl')
    
    # 可视化结果
    plot_training_results(model_data, y, y_pred_final)
    
    print("模型训练完成并保存!")
    return model_data

def plot_training_results(model_data, y_true, y_pred):
    """绘制训练结果可视化"""
    plt.figure(figsize=(15, 10))
    
    # 1. 交叉验证分数
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(model_data['cv_scores']) + 1), model_data['cv_scores'], 'bo-')
    plt.xlabel('折数')
    plt.ylabel('加权F1分数')
    plt.title('交叉验证性能')
    plt.grid(True)
    
    # 2. 混淆矩阵
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 3. 特征重要性
    plt.subplot(2, 2, 3)
    top_features = model_data['feature_importance'].head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('重要性')
    plt.title('Top 10 特征重要性')
    plt.gca().invert_yaxis()
    
    # 4. 类别分布
    plt.subplot(2, 2, 4)
    class_dist = model_data['class_distribution']
    plt.bar(class_dist.keys(), class_dist.values())
    plt.xlabel('类别')
    plt.ylabel('样本数')
    plt.title('类别分布')
    
    plt.tight_layout()
    plt.savefig('./models/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 可选：专门的类别平衡函数
def apply_class_balancing(X_train, y_train, method='smote'):
    """应用类别平衡技术"""
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)
    elif method == 'random_oversample':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X_train, y_train)
    else:
        return X_train, y_train

# 使用示例
if __name__ == "__main__":
    # 执行特征工程
    result = feature_engineering_pipeline('your_data.csv')
    
    # 使用处理后的特征训练模型
    train_model_with_engineered_features('./features/engineered_features.pkl')