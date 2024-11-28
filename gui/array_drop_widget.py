import json
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal


class ArrayFileDropWidget(QWidget):
    """Widget that accepts drag and drop of JSON/TXT files containing arrays"""

    array_loaded_signal = pyqtSignal(list)  # Signal emitted when array is loaded

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setObjectName("arrayDropLabel")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(
            "📄 Drop your phi offset array file\n(Accepts .txt files, see follows)\n `1,2,4,11` in your file."
        )
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setMinimumHeight(120)
        self.label.setProperty("dragOver", False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.label.setProperty("dragOver", True)
            self.style().unpolish(self.label)
            self.style().polish(self.label)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.label.setProperty("dragOver", False)
        self.style().unpolish(self.label)
        self.style().polish(self.label)

    def dropEvent(self, event):
        self.label.setProperty("dragOver", False)
        self.style().unpolish(self.label)
        self.style().polish(self.label)

        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            if file_path.endswith(".txt"):
                try:
                    with open(file_path, "r") as f:
                        content = f.read().strip()
                        # Split by comma and convert to float list
                        data = [float(x) for x in content.split(",")]

                    if isinstance(data, list):
                        self.array_loaded_signal.emit(data)
                        self.label.setText(
                            f"✅ Successfully loaded array\nElements: {len(data)}\n\nDrop another file to replace"
                        )
                    else:
                        self.label.setText(
                            "❌ Error: File must contain comma-separated numbers\n\nTry dropping another file"
                        )
                except Exception as e:
                    self.label.setText(
                        f"❌ Error loading file:\n{str(e)}\n\nTry dropping another file"
                    )
            else:
                self.label.setText(
                    "❌ Please drop a TXT file\n\nTry dropping another file"
                )
