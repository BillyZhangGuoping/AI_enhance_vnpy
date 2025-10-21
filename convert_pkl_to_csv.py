import pickle
import pandas as pd

# 读取 .pkl 文件
with open('features/fu2601_1m_featured.pkl', 'rb') as f:
    data = pickle.load(f)

# 转换为 DataFrame 并保存为 .csv
if isinstance(data, (list, dict)):
    df = pd.DataFrame(data)
else:
    df = pd.DataFrame([data])  # 处理单条数据

df.to_csv('features/fu2601_1m_featured.csv', index=False)
print("文件已成功转换为 features/fu2601_1m_featured.csv")