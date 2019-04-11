from PyQt5.QtWidgets import QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
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
        self.rho = QSlider(Qt.Horizontal)
        self.theta = QSlider(Qt.Horizontal)

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
        rho_label = QLabel("Rho Error (Distance): ")
        theta_label = QLabel("Theta Error (Angle): ")

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

        self.rho.setMinimum(1)
        self.rho.setMaximum(min(self.image.image.shape[:2]))
        self.rho.setValue(50)
        self.rho.setTickInterval(1)
        self.rho.setTickPosition(QSlider.TicksBelow)
        self.rho.valueChanged.connect(self.on_value_change)

        self.theta.setMinimum(1)
        self.theta.setMaximum(180)
        self.theta.setValue(36)
        self.theta.setTickInterval(1)
        self.theta.setTickPosition(QSlider.TicksBelow)
        self.theta.valueChanged.connect(self.on_value_change)

        menu_layout.addWidget(thresh_max_label)
        menu_layout.addWidget(self.thresh_max)
        menu_layout.addWidget(kernel_label)
        menu_layout.addWidget(self.kernel)
        menu_layout.addWidget(iter_label)
        menu_layout.addWidget(self.iteration)
        menu_layout.addWidget(rho_label)
        menu_layout.addWidget(self.rho)
        menu_layout.addWidget(theta_label)
        menu_layout.addWidget(self.theta)



    # Event Handlers
    def on_value_change(self):
        thresh = self.thresh_max.value()
        kernel = self.kernel.value()
        iteration = self.iteration.value()
        rho = self.rho.value()
        theta = self.theta.value()
        if kernel % 2 == 0:
            kernel -= 1

        self.image.adaptive_thresholding(thresh, kernel, iteration)
        self.image.calculate_convex_hull()
        self.image.mask_with_convex_hull()
        self.image.make_hough_transformation(rho_value=rho, theta_value=theta)
        self.image.contouring()
        self.image.component_calculation()

        result_image = QPixmap(Image.image_cv2qt(self.image.test_image))
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
