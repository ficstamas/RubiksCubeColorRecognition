from PyQt5.QtWidgets import QWidget, QFileDialog, QVBoxLayout


class FileDialog:
    def __init__(self, parent):
        super(FileDialog, self).__init__()
        self.parent = parent
        self.file = None
        self.initUI()

    def initUI(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.parent, "Open Image", "",
                                                  "PNG Files (*.png);;JPG Files (*.jpg)", options=options)
        self.file = fileName