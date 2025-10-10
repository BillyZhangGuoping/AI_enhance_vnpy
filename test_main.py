import sys
from unittest.mock import MagicMock
import pytest
from PySide6.QtWidgets import QApplication, QLabel
from main import PrintRedirector, MainWindow

# Test PrintRedirector class
class TestPrintRedirector:
    """Test cases for the PrintRedirector class."""

    def test_write_method(self):
        """Test the write method of PrintRedirector.
        Verifies that text is correctly appended to the label."""
        label = QLabel()
        redirector = PrintRedirector(label)
        redirector.write("Test text")
        assert label.text() == "Test text"

    def test_write_method_multiple_calls(self):
        """Test multiple calls to the write method.
        Verifies that text is correctly accumulated."""
        label = QLabel()
        redirector = PrintRedirector(label)
        redirector.write("First ")
        redirector.write("Second ")
        redirector.write("Third")
        assert label.text() == "First Second Third"

    def test_flush_method(self):
        """Test the flush method of PrintRedirector.
        Verifies that the method exists and does nothing."""
        label = QLabel()
        redirector = PrintRedirector(label)
        redirector.flush()  # Should not raise any exceptions

# Test MainWindow class
class TestMainWindow:
    """Test cases for the MainWindow class."""

    @pytest.fixture
    def app(self):
        """Fixture to provide a QApplication instance."""
        app = QApplication(sys.argv)
        yield app
        app.quit()

    def test_init_method(self, app):
        """Test the initialization of MainWindow.
        Verifies that the window is correctly set up with a label."""
        window = MainWindow()
        assert window.windowTitle() == "Print Output"
        assert window.label.text() == "Print output will appear here:"

    def test_print_redirection(self, app):
        """Test the print redirection functionality.
        Verifies that print statements are redirected to the label."""
        window = MainWindow()
        print("Test print")
        assert "Test print" in window.label.text()

    def test_multiple_prints(self, app):
        """Test multiple print statements.
        Verifies that all prints are correctly redirected."""
        window = MainWindow()
        print("First print")
        print("Second print")
        assert "First print" in window.label.text()
        assert "Second print" in window.label.text()

    def test_empty_print(self, app):
        """Test an empty print statement.
        Verifies that the label remains unchanged."""
        window = MainWindow()
        initial_text = window.label.text()
        print("")
        assert window.label.text() == initial_text
