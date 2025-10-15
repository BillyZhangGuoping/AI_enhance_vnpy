import sys
from PySide6.QtWidgets import QApplication
from UI.main_UI import ProcessDataUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessDataUI(parent=None)
    window.show()
    sys.exit(app.exec())