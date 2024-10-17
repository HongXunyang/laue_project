import sys
from PyQt5.QtWidgets import QApplication
from gui import MainWindow


def main():
    app = QApplication(sys.argv)

    # Load and apply the style sheet
    with open("config/gui_styles.css", "r") as f:
        app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
