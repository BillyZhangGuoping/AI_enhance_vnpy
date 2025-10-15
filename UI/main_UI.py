import sys
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PySide6.QtCore import Qt
import pandas as pd
import os

class ProcessDataUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Processing")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        self.label = QLabel("Select a CSV file to process:")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.select_button = QPushButton("Select CSV File")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        self.csv_button = QPushButton("Output CSV")
        self.csv_button.clicked.connect(self.save_as_csv)
        self.csv_button.hide()
        layout.addWidget(self.csv_button)

        self.parquet_button = QPushButton("Output Parquet")
        self.parquet_button.clicked.connect(self.save_as_parquet)
        self.parquet_button.hide()
        layout.addWidget(self.parquet_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_ui)
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        self.df = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.df = pd.read_csv(file_path)
            self.label.setText(f"File loaded: {os.path.basename(file_path)}")
            self.csv_button.show()
            self.parquet_button.show()

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

    def close_ui(self):
        self.hide()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Interface")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        self.data_button = QPushButton("数据整理")
        self.data_button.clicked.connect(self.show_process_data_ui)
        layout.addWidget(self.data_button)

        self.setLayout(layout)

        self.process_data_ui = ProcessDataUI(self)

    def show_process_data_ui(self):
        sys.stdout.write("Button clicked: Showing process data UI\n")  # Direct debug output
        self.hide()
        self.process_data_ui.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())