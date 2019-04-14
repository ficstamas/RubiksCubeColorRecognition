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
        self.canny_thresh_min = QSlider(Qt.Horizontal)
        self.canny_thresh_max = QSlider(Qt.Horizontal)
        self.hlml = QSlider(Qt.Horizontal)
        self.hlmg = QSlider(Qt.Horizontal)
        self.theta = QSlider(Qt.Horizontal)
        self.bl_slider = QSlider(Qt.Horizontal)
        self.cb_slider = QSlider(Qt.Horizontal)

        self._prepare_ui()

    def _prepare_ui(self):
        for i in reversed(range(self.layout.count())):
            if self.layout.itemAt(i).widget() is not None:
                self.layout.itemAt(i).widget().setParent(None)

        self.image.detect_colors()
        result_image = QPixmap(Image.image_cv2qt(self.image.result_image))
        result_image = result_image.scaled(300, 300, Qt.KeepAspectRatio)
        self.result_image(result_image)
        self.set_result_image()

        # Add MenuItems
        menu_layout = self.layout

        # Labels
        canny_thresh_min_label = QLabel("Canny Thresh Min Value: ")
        canny_thresh_max_label = QLabel("Canny Thresh Max Value: ")
        hlml_label = QLabel("Hough Lines - Min Length of Line: ")
        hlmg_label = QLabel("Hough Lines - Max Gap Between Lines: ")
        theta_label = QLabel("Theta Error (Angle): ")
        bl_label = QLabel("Selected Lines: ")
        cb_label = QLabel("Color Balance (Percent): ")

        # Sliders
        self.canny_thresh_min.setMinimum(0)
        self.canny_thresh_min.setMaximum(255)
        self.canny_thresh_min.setValue(50)
        self.canny_thresh_min.setTickInterval(1)
        self.canny_thresh_min.setTickPosition(QSlider.TicksBelow)
        self.canny_thresh_min.valueChanged.connect(self.on_value_change)

        self.canny_thresh_max.setMinimum(0)
        self.canny_thresh_max.setMaximum(255)
        self.canny_thresh_max.setValue(200)
        self.canny_thresh_max.setTickInterval(2)
        self.canny_thresh_max.setTickPosition(QSlider.TicksBelow)
        self.canny_thresh_max.valueChanged.connect(self.on_value_change)

        self.hlml.setMinimum(0)
        self.hlml.setMaximum(300)
        self.hlml.setValue(15)
        self.hlml.setTickInterval(1)
        self.hlml.setTickPosition(QSlider.TicksBelow)
        self.hlml.valueChanged.connect(self.on_value_change)

        self.hlmg.setMinimum(1)
        self.hlmg.setMaximum(200)
        self.hlmg.setValue(20)
        self.hlmg.setTickInterval(1)
        self.hlmg.setTickPosition(QSlider.TicksBelow)
        self.hlmg.valueChanged.connect(self.on_value_change)

        self.theta.setMinimum(1)
        self.theta.setMaximum(90)
        self.theta.setValue(30)
        self.theta.setTickInterval(1)
        self.theta.setTickPosition(QSlider.TicksBelow)
        self.theta.valueChanged.connect(self.on_value_change)

        self.bl_slider.setMinimum(3)
        self.bl_slider.setMaximum(90)
        self.bl_slider.setValue(12)
        self.bl_slider.setTickInterval(1)
        self.bl_slider.setTickPosition(QSlider.TicksBelow)
        self.bl_slider.valueChanged.connect(self.on_value_change)

        self.cb_slider.setMinimum(0)
        self.cb_slider.setMaximum(99)
        self.cb_slider.setValue(1)
        self.cb_slider.setTickInterval(1)
        self.cb_slider.setTickPosition(QSlider.TicksBelow)
        self.cb_slider.valueChanged.connect(self.on_value_change)

        menu_layout.addWidget(canny_thresh_min_label)
        menu_layout.addWidget(self.canny_thresh_min)
        menu_layout.addWidget(canny_thresh_max_label)
        menu_layout.addWidget(self.canny_thresh_max)
        menu_layout.addWidget(hlml_label)
        menu_layout.addWidget(self.hlml)
        menu_layout.addWidget(hlmg_label)
        menu_layout.addWidget(self.hlmg)
        menu_layout.addWidget(theta_label)
        menu_layout.addWidget(self.theta)
        menu_layout.addWidget(bl_label)
        menu_layout.addWidget(self.bl_slider)
        menu_layout.addWidget(cb_label)
        menu_layout.addWidget(self.cb_slider)



    # Event Handlers
    def on_value_change(self):
        cmin = self.canny_thresh_min.value()
        cmax = self.canny_thresh_max.value()
        hlml = self.hlml.value()
        hlmg = self.hlmg.value()
        theta = self.theta.value()
        bl = self.bl_slider.value()
        cb = self.cb_slider.value()

        self.image.detect_colors(theta_error=theta, canny_threshold1=cmin, canny_threshold2=cmax,
                                 hlines_min_length=hlml, hlines_max_gap=hlmg, no_best_lines=bl, color_balance_value=cb)
        result_image = QPixmap(Image.image_cv2qt(self.image.result_image))
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
