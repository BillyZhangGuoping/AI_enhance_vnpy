import tkinter as tk
from tkinter import filedialog
import pandas as pd
import pickle

class FeatureEngineerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Engineer UI")
        
        # 选择 Parquet 文件按钮
        self.select_parquet_button = tk.Button(
            root, text="选择 Parquet 文件", command=self.select_parquet_file
        )
        self.select_parquet_button.pack(pady=10)
        
        # 处理按钮
        self.process_button = tk.Button(
            root, text="处理数据", command=self.process_data
        )
        self.process_button.pack(pady=10)
        
        # 保存 PKL 文件按钮
        self.save_pkl_button = tk.Button(
            root, text="保存 PKL 文件", command=self.save_pkl_file
        )
        self.save_pkl_button.pack(pady=10)
        
        # 状态标签
        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=10)
        
        # 初始化变量
        self.parquet_file_path = None
        self.processed_data = None
    
    def select_parquet_file(self):
        """选择 Parquet 文件"""
        self.parquet_file_path = filedialog.askopenfilename(
            title="选择 Parquet 文件",
            filetypes=[("Parquet 文件", "*.parquet")]
        )
        if self.parquet_file_path:
            self.status_label.config(text=f"已选择文件: {self.parquet_file_path}")
    
    def process_data(self):
        """处理数据"""
        if not self.parquet_file_path:
            self.status_label.config(text="请先选择 Parquet 文件")
            return
        
        try:
            # 读取 Parquet 文件
            df = pd.read_parquet(self.parquet_file_path)
            
            # 示例处理逻辑（占位符，后续可扩展）
            self.processed_data = df.to_dict()
            
            self.status_label.config(text="数据处理完成")
        except Exception as e:
            self.status_label.config(text=f"处理失败: {str(e)}")
    
    def save_pkl_file(self):
        """保存 PKL 文件"""
        if not self.processed_data:
            self.status_label.config(text="请先处理数据")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="保存 PKL 文件",
            defaultextension=".pkl",
            filetypes=[("PKL 文件", "*.pkl")]
        )
        
        if save_path:
            try:
                with open(save_path, "wb") as f:
                    pickle.dump(self.processed_data, f)
                self.status_label.config(text=f"文件已保存到: {save_path}")
            except Exception as e:
                self.status_label.config(text=f"保存失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureEngineerUI(root)
    root.mainloop()