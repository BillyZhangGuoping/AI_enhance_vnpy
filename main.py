import sys
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

class PrintRedirector:
    def __init__(self, label):
        self.label = label

    def write(self, text):
        self.label.setText(self.label.text() + text)

    def flush(self):
        pass

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Print Output")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        self.label = QLabel("Print output will appear here:")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Redirect print output to the label
        sys.stdout = PrintRedirector(self.label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Example print statements
    print("Hello, PySide!")
    print("This is a test output.")

    sys.exit(app.exec())