import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from PySide6.QtCore import Qt
import pandas as pd
import os
import pickle

# 从 Processor 模块导入方法
from Processor.process_kline_data import clean_data, add_technical_indicators, add_time_features, add_target_variable


class ProcessDataUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Processing")
        layout = QVBoxLayout()
        self.label = QLabel("Select a CSV file to process:")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.select_button = QPushButton("Select CSV File")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        self.process_button = QPushButton("处理数据")
        self.process_button.clicked.connect(self.process_data)
        layout.addWidget(self.process_button)

        self.csv_button = QPushButton("Output CSV")
        self.csv_button.clicked.connect(self.save_as_csv)
        self.csv_button.hide()
        layout.addWidget(self.csv_button)

        self.parquet_button = QPushButton("Output Parquet")
        self.parquet_button.clicked.connect(self.save_as_parquet)
        self.parquet_button.hide()
        layout.addWidget(self.parquet_button)

        self.setLayout(layout)
        self.df = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.df = pd.read_csv(file_path)
            file_size = os.path.getsize(file_path) / 1024  # KB
            self.label.setText(f"File loaded: {os.path.basename(file_path)} (Size: {file_size:.2f} KB, Rows: {len(self.df)})")
            self.csv_button.show()
            self.parquet_button.show()

    def process_data(self):
        if self.df is None:
            self.label.setText("未读取数据 CSV 文件")
            return

        try:
            self.df = clean_data(self.df)
            self.df = add_technical_indicators(self.df)
            self.df = add_time_features(self.df)
            self.df = add_target_variable(self.df)
            self.label.setText("数据处理完成")
        except Exception as e:
            self.label.setText(f"处理失败: {str(e)}")

    def save_as_csv(self):
        if self.df is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
            if save_path:
                self.df.to_csv(save_path, index=False)
                self.label.setText(f"CSV saved to {save_path}")

    def save_as_parquet(self):
        if self.df is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Parquet File", "", "Parquet Files (*.parquet)")
            if save_path:
                self.df.to_parquet(save_path, engine='pyarrow')
                self.label.setText(f"Parquet saved to {save_path}")

class FeatureEngineerUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Engineer")
        layout = QVBoxLayout()
        self.select_button = QPushButton("选择 Parquet 文件")
        self.select_button.clicked.connect(self.select_parquet_file)
        layout.addWidget(self.select_button)

        self.process_button = QPushButton("处理数据")
        self.process_button.clicked.connect(self.process_data)
        layout.addWidget(self.process_button)

        self.save_button = QPushButton("保存 PKL 文件")
        self.save_button.clicked.connect(self.save_pkl_file)
        layout.addWidget(self.save_button)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.parquet_file_path = None
        self.processed_data = None

    def select_parquet_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 Parquet 文件", "", "Parquet Files (*.parquet)")
        if file_path:
            self.parquet_file_path = file_path
            self.status_label.setText(f"已选择文件: {os.path.basename(file_path)}")

    def process_data(self):
        if not self.parquet_file_path:
            self.status_label.setText("请先选择 Parquet 文件")
            return

        try:
            df = pd.read_parquet(self.parquet_file_path)
            self.processed_data = df.to_dict()
            self.status_label.setText("数据处理完成")
        except Exception as e:
            self.status_label.setText(f"处理失败: {str(e)}")

    def save_pkl_file(self):
        if not self.processed_data:
            self.status_label.setText("请先处理数据")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "保存 PKL 文件", "", "PKL Files (*.pkl)")
        if save_path:
            try:
                with open(save_path, "wb") as f:
                    pickle.dump(self.processed_data, f)
                self.status_label.setText(f"文件已保存到: {save_path}")
            except Exception as e:
                self.status_label.setText(f"保存失败: {str(e)}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Interface")
        self.setGeometry(100, 100, 800, 400)

        # 使用 QHBoxLayout 将主界面分为两个区域
        layout = QHBoxLayout()

        # 区域1：ProcessDataUI
        self.process_data_ui = ProcessDataUI()
        layout.addWidget(self.process_data_ui)

        # 区域2：FeatureEngineerUI
        self.feature_engineer_ui = FeatureEngineerUI()
        layout.addWidget(self.feature_engineer_ui)

        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())