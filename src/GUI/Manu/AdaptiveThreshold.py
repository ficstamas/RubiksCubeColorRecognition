from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QSlider, QLineEdit, QFormLayout
from src.GUI.FileDialog import FileDialog
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QIntValidator, QFont
from src.Model.Image import Image


class AdaptiveThreshold:
    def __init__(self, image: Image, result_image, set_result_image):
        self.__image = image
        self.__layout = QVBoxLayout()

        self.result_image = result_image
        self.set_result_image = set_result_image

        # Sliders
        self.thresh_min = QSlider(Qt.Horizontal)
        self.thresh_max = QSlider(Qt.Horizontal)
        self.kernel = QSlider(Qt.Horizontal)
        self.iteration = QSlider(Qt.Horizontal)
        self.max_gap = QSlider(Qt.Horizontal)
        self.min_length = QSlider(Qt.Horizontal)

        self._prepare_ui()

    def _prepare_ui(self):
        for i in reversed(range(self.layout.count())):
            if self.layout.itemAt(i).widget() is not None:
                self.layout.itemAt(i).widget().setParent(None)

        self.image.adaptive_thresholding(255, 15)
        result_image = QPixmap(Image.image_cv2qt(self.image.adaptive_threshold))
        result_image = result_image.scaled(300, 300, Qt.KeepAspectRatio)
        self.result_image(result_image)
        self.set_result_image()

        # Add MenuItems
        menu_layout = self.layout

        # Labels
        thresh_max_label = QLabel("Thresh Value: ")
        kernel_label = QLabel("Kernel Size: ")
        iter_label = QLabel("Iterations: ")
        gap_label = QLabel("Maximum Line Gap: ")
        length_label = QLabel("Minimum Line Length: ")

        # Sliders
        self.thresh_max.setMinimum(0)
        self.thresh_max.setMaximum(255)
        self.thresh_max.setValue(255)
        self.thresh_max.setTickInterval(1)
        self.thresh_max.setTickPosition(QSlider.TicksBelow)
        self.thresh_max.valueChanged.connect(self.on_value_change)

        self.kernel.setMinimum(3)
        self.kernel.setMaximum(51)
        self.kernel.setValue(15)
        self.kernel.setTickInterval(2)
        self.kernel.setTickPosition(QSlider.TicksBelow)
        self.kernel.valueChanged.connect(self.on_value_change)

        self.iteration.setMinimum(0)
        self.iteration.setMaximum(12)
        self.iteration.setValue(2)
        self.iteration.setTickInterval(1)
        self.iteration.setTickPosition(QSlider.TicksBelow)
        self.iteration.valueChanged.connect(self.on_value_change)

        self.max_gap.setMinimum(10)
        self.max_gap.setMaximum(100)
        self.max_gap.setValue(50)
        self.max_gap.setTickInterval(1)
        self.max_gap.setTickPosition(QSlider.TicksBelow)
        self.max_gap.valueChanged.connect(self.on_value_change)

        self.min_length.setMinimum(30)
        self.min_length.setMaximum(120)
        self.min_length.setValue(50)
        self.min_length.setTickInterval(1)
        self.min_length.setTickPosition(QSlider.TicksBelow)
        self.min_length.valueChanged.connect(self.on_value_change)

        menu_layout.addWidget(thresh_max_label)
        menu_layout.addWidget(self.thresh_max)
        menu_layout.addWidget(kernel_label)
        menu_layout.addWidget(self.kernel)
        menu_layout.addWidget(iter_label)
        menu_layout.addWidget(self.iteration)
        menu_layout.addWidget(gap_label)
        menu_layout.addWidget(self.max_gap)
        menu_layout.addWidget(length_label)
        menu_layout.addWidget(self.min_length)



    # Event Handlers
    def on_value_change(self):
        thresh = self.thresh_max.value()
        kernel = self.kernel.value()
        iteration = self.iteration.value()
        max_gap = self.max_gap.value()
        min_length = self.min_length.value()
        if kernel % 2 == 0:
            kernel -= 1

        self.image.adaptive_thresholding(thresh, kernel, iteration)
        self.image.calculate_convex_hull()
        self.image.mask_with_convex_hull()
        self.image.make_canny(max_gap=max_gap, min_length=min_length)
        self.image.contouring()
        self.image.component_calculation()
        self.image.contour_bounding_box()

        result_image = QPixmap(Image.image_cv2qt(self.image.bounding_box))
        result_image = result_image.scaled(300, 300, Qt.KeepAspectRatio)
        self.result_image(result_image)
        self.set_result_image()

    @property
    def layout(self):
        return self.__layout

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, img: image):
        self.__image = img
