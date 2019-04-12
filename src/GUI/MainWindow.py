from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QSlider, QLineEdit, QFormLayout
from src.GUI.Elements.FileDialog import FileDialog
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QIntValidator, QFont
from src.Model.Image import Image
from src.GUI.Elements.AdaptiveThreshold import AdaptiveThreshold


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.image = Image()

        # Images
        self.loaded_image_holder = QLabel()
        self.loaded_image_holder.setGeometry(0, 0, 300, 300)
        self.loaded_image_holder.resize(300, 300)
        self.loaded_image_holder.setPixmap(QPixmap())

        self.__result_image = QPixmap()
        self.result_image_label = QLabel()
        self.result_image_label.setGeometry(0, 0, 300, 300)
        self.result_image_label.resize(300, 300)
        self.result_image_label.setPixmap(self.result_image)

        # Labels
        self.path = QLabel("")

        # Sliders
        self.thresh_min = QSlider(Qt.Horizontal)
        self.thresh_max = QSlider(Qt.Horizontal)
        self.kernel = QSlider(Qt.Horizontal)

        # Input Fields
        self.iteration_field = QLineEdit()

        # Layouts
        self.menu_layout = QVBoxLayout()
        self.menu_load_layout = QVBoxLayout()
        self.menu_separator_layout = QVBoxLayout()
        self.menu_debug_layout = QVBoxLayout()
        self.image_layout = QVBoxLayout()
        self.global_layout = QHBoxLayout()

        # Special Layout
        self.adaptive_layout_class = None

        self.initUI()

    def initUI(self):
        self.setGeometry(0, 0, 800, 600)
        self.setWindowTitle('App')

        # Layouts
        load_img_button = QPushButton('Load Image')

        debug_button = QPushButton('Debug')
        debug_button.clicked.connect(self.on_click_debug)
        self.menu_debug_layout.addWidget(debug_button)

        # Image Layout (Right Column)
        self.image_layout.addWidget(self.loaded_image_holder)
        self.image_layout.addWidget(self.result_image_label)

        # Menu Layout (Left Column)
        self.menu_load_layout.addWidget(self.path)
        self.menu_load_layout.addWidget(load_img_button)

        self.menu_layout.addLayout(self.menu_load_layout)
        self.menu_layout.addLayout(self.menu_separator_layout)
        self.menu_layout.addLayout(self.menu_debug_layout)

        self.global_layout.addLayout(self.image_layout)
        self.global_layout.addLayout(self.menu_layout)
        self.setLayout(self.global_layout)

        load_img_button.clicked.connect(self.open_file_handler)

        self.show()

    @pyqtSlot()
    def open_file_handler(self):
        file_handler = FileDialog(self)
        self.file_path = file_handler.file
        self.path.setText(self.file_path)

        self.image = Image()

        self.image.imread(self.file_path)

        img = QPixmap(Image.image_cv2qt(self.image.image))
        img = img.scaled(300, 300, Qt.KeepAspectRatio)
        self.loaded_image_holder.setPixmap(img)


        # self.image.make_canny()
        self.result_image = QPixmap(Image.image_cv2qt(self.image.result_image))
        self.result_image = self.result_image.scaled(300, 300, Qt.KeepAspectRatio)
        self.set_result_image()

        if self.adaptive_layout_class is None:
            self.adaptive_layout_class = AdaptiveThreshold(self.image, self.set_result_image_f, self.set_result_image)
        self.adaptive_layout_class.image = self.image

        self.menu_separator_layout = self.adaptive_layout_class.layout
        self.reload_menu()

    def create_menu(self):
        for i in reversed(range(self.menu_separator_layout.count())):
            if self.menu_separator_layout.itemAt(i).widget() is not None:
                self.menu_separator_layout.itemAt(i).widget().setParent(None)

        # # Buttons
        # load_img_button = QPushButton('Load Image')
        # load_img_button.clicked.connect(self.open_file_handler)

        # Add MenuItems
        menu_layout = self.menu_separator_layout

        # Labels
        thresh_min_label = QLabel("Thresh Min Value: ")
        thresh_max_label = QLabel("Thresh Max Value: ")
        kernel_label = QLabel("Kernel Size: ")

        # Sliders
        self.thresh_min.setMinimum(0)
        self.thresh_min.setMaximum(255)
        self.thresh_min.setValue(30)
        self.thresh_min.setTickInterval(1)
        self.thresh_min.setTickPosition(QSlider.TicksBelow)
        self.thresh_min.valueChanged.connect(self.on_value_change)

        self.thresh_max.setMinimum(30)
        self.thresh_max.setMaximum(255)
        self.thresh_max.setValue(255)
        self.thresh_max.setTickInterval(1)
        self.thresh_max.setTickPosition(QSlider.TicksBelow)
        self.thresh_max.valueChanged.connect(self.on_value_change)

        self.kernel.setMinimum(4)
        self.kernel.setMaximum(32)
        self.kernel.setValue(6)
        self.kernel.setTickInterval(1)
        self.kernel.setTickPosition(QSlider.TicksBelow)
        self.kernel.valueChanged.connect(self.on_value_change)

        # Input Fields
        self.iteration_field.setValidator(QIntValidator())
        self.iteration_field.setMaxLength(2)
        self.iteration_field.setAlignment(Qt.AlignRight)
        self.iteration_field.setFont(QFont("Arial", 13))
        self.iteration_field.setText("3")
        self.iteration_field.textChanged.connect(self.on_value_change)

        flo = QFormLayout()
        flo.addRow("Iterations", self.iteration_field)

        # menu_layout.addWidget(self.path)
        # menu_layout.addWidget(load_img_button)
        menu_layout.addWidget(thresh_min_label)
        menu_layout.addWidget(self.thresh_min)
        menu_layout.addWidget(thresh_max_label)
        menu_layout.addWidget(self.thresh_max)
        menu_layout.addWidget(kernel_label)
        menu_layout.addWidget(self.kernel)
        menu_layout.addLayout(flo)

    def set_result_image(self):
        self.result_image_label.setPixmap(self.result_image)

    def reload_menu(self):
        for i in reversed(range(self.menu_layout.count())):
            if self.menu_layout.itemAt(i).widget() is not None:
                self.menu_layout.itemAt(i).widget().setParent(None)

        self.menu_layout.addLayout(self.menu_load_layout)
        self.menu_layout.addLayout(self.menu_separator_layout)
        self.menu_layout.addLayout(self.menu_debug_layout)

    # Event Handlers
    def on_value_change(self):
        val = self.thresh_min.value()
        val2 = self.thresh_max.value()
        val3 = self.kernel.value()
        val4 = 1
        try:
            val4 = int(self.iteration_field.text())
        except Exception:
            val4 = 1

        self.thresh_max.setMinimum(val)
        self.thresh_min.setMaximum(val2)

        self.result_image = QPixmap(Image.image_cv2qt(self.image.image))
        self.result_image = self.result_image.scaled(300, 300, Qt.KeepAspectRatio)
        self.set_result_image()

    @pyqtSlot()
    def on_click_debug(self):
        # self.image.debug()
        pass

    @property
    def result_image(self) -> QPixmap:
        return self.__result_image

    @result_image.setter
    def result_image(self, img: QPixmap):
        self.__result_image = img

    def set_result_image_f(self, img: QPixmap):
        self.result_image = img

