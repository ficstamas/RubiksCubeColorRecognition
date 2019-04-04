import cv2 as cv
import numpy as np
import copy
from PyQt5.QtGui import QImage
import asyncio
import math


class Image:
    def __init__(self):
        self.__image = None
        self.__image_gray = None
        self.__contour = None
        self.__mask = None
        self.__masked_image = None
        self.__component_image = None
        self.__hough_lines = None
        self.__adaptive_threshold = None
        self.__convex_hull = None
        self.__test = None
        self.__recolored_image = None
        self.__ch_masked = None
        self.__contour_data = None
        self.__bb = None
        self.__convex_hull_outline = None
        self.__color_pallet = {
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'blue': [0, 93, 167],
            'green': [0, 122, 30],
            'red': [206, 30, 33],
            'orange': [255, 96, 2],
            'yellow': [237, 190, 4]
        }

    def imread(self, path: str):
        self.__image = cv.imread(path)
        self.__image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # self.recolor_image()

    def adaptive_thresholding(self, thresh, kernel, iteration=3):
        neg = cv.bitwise_not(self.image_gray)
        self.__adaptive_threshold = cv.adaptiveThreshold(neg, thresh, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, kernel, -2)
        kernel_matrix = np.ones((3, 3), np.uint8)
        self.__adaptive_threshold = cv.erode(self.adaptive_threshold, kernel_matrix, iterations=iteration)
        self.__adaptive_threshold = cv.morphologyEx(self.adaptive_threshold, cv.MORPH_OPEN, kernel_matrix)

    @classmethod
    async def find_white_on_mask(cls, y, row):
        points = []
        for x, color in enumerate(row):
            if color > 0:
                points.append([x*4, y*4])
        return points

    def calculate_convex_hull(self):
        points = []
        loop = asyncio.get_event_loop()
        coro = asyncio.gather(*[self.find_white_on_mask(y, row) for y, row in enumerate(self.adaptive_threshold[::4, ::4])])
        result = loop.run_until_complete(coro)
        for r in result:
            points += r

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        convex_hull = np.array([lower[:-1] + upper[:-1]])

        outline = np.zeros((self.adaptive_threshold.shape[0], self.adaptive_threshold.shape[1], 3), dtype=np.uint8)
        drawing = np.zeros((self.adaptive_threshold.shape[0], self.adaptive_threshold.shape[1], 3), dtype=np.uint8)
        if convex_hull.shape[1] > 0:
            for i in range(len(convex_hull)):
                color = (255, 255, 255)
                # cv.drawContours(drawing, contours, i, color)
                cv.drawContours(drawing, convex_hull, -1, color, thickness=-1)
                cv.drawContours(outline, convex_hull, -1, color, thickness=3)

        self.__convex_hull = drawing
        self.__convex_hull_outline = outline

    def mask_with_convex_hull(self):
        img = self.image.copy()
        mask = cv.cvtColor(self.convex_hull, cv.COLOR_BGR2GRAY)
        b, g, r = cv.split(img)
        r = cv.bitwise_and(r, mask)
        g = cv.bitwise_and(g, mask)
        b = cv.bitwise_and(b, mask)
        self.__ch_masked = cv.merge((b, g, r))

    @classmethod
    def _distance(cls, coord1: list, coord2: list) -> float:
        d = 0
        for i in range(coord1.__len__()):
            d += (coord1[i] - coord2[i])**2
        return np.sqrt(d)

    def debug(self):
        cv.imshow("Colored Image", self.image)
        cv.imshow("Adaptive Threshold Image", self.adaptive_threshold)
        cv.imshow("Convex Hull", self.__convex_hull)
        cv.imshow("Hough Lines", self.hough_lines)
        cv.imshow("Contour", self.contour)
        cv.imshow("Component Map", self.component_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_hough_transformation(self, thresh_min=0, thresh_max=255, rho_value=50, theta_value=50):
        # Creating Canny image
        canny = cv.Canny(self.__ch_masked, thresh_min, thresh_max)
        # self.__canny = canny
        # Setting up the result image
        drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
        # Calculating Hough Lines from the Canny image
        lines = cv.HoughLines(canny, 1, np.pi / 180, 100, None)

        # Converting Polar coordinates to Cartesian
        def polar_to_cartesian(rho, theta):
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            return pt1, pt2

        # TODO make function which will represent negative negative numbers as pi-|number| and vica verse
        def is_theta_error_in_range(theta):
            pass

        line_classes = []

        theta_error = np.pi/(180/theta_value)
        rho_error = rho_value
        # Filtering out Lines and organizing them into classes by Rho
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]

                pt1, pt2 = polar_to_cartesian(rho, theta)
                if line_classes.__len__() == 0:
                    line_classes.append([theta, ])
                is_theta_exists = False
                for k, l in enumerate(line_classes):
                    lower_bound = l[0]-theta_error
                    upper_bound = l[0]+theta_error
                    if lower_bound < 0:

                    if lower_bound <= theta < upper_bound:
                        is_rho_fine = True
                        for t in l[1:]:
                            if t[2]-rho_error < rho < t[2]+rho_error:
                                is_rho_fine = False
                                break
                        if is_rho_fine:
                            line_classes[k].append((pt1, pt2, rho))
                        is_theta_exists = True
                        break

                if not is_theta_exists:
                    line_classes.append([theta, (pt1, pt2, rho)])

        masked_lines = []
        # Masking lines and setting endpoints according to mask
        # for class_ in line_classes:
        #     new_class = []
        #     for points in class_[1:]:
        #         print(points)

        # Drawing lines on result image
        for class_ in line_classes[1:3]:
            for data in class_[1:]:
                pt1 = data[0]
                pt2 = data[1]
                cv.line(drawing, pt1, pt2, (255, 255, 255), 3, cv.LINE_AA)

        self.__hough_lines = drawing

    def contouring(self):
        canny = cv.cvtColor(self.hough_lines, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.__contour_data = contours
        drawing = np.zeros((self.hough_lines.shape[0], self.hough_lines.shape[1], 3), dtype=np.uint8)

        cv.drawContours(drawing, contours, -1, (255, 255, 255), 3)
        self.__contour = drawing

    def component_calculation(self):
        cont = cv.cvtColor(self.__contour, cv.COLOR_BGR2GRAY)
        cont = cv.bitwise_not(cont)
        retVal, labels = cv.connectedComponents(cont, None, 8, cv.CV_16U)
        labelsNorm = cv.normalize(labels, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
        self.__component_image = labelsNorm

    @classmethod
    def image_cv2qt(cls, img) -> QImage:
        imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, channel = imgrgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(imgrgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

        return qImg

    def produce_test_image(self):
        self.__test = cv.normalize(
            cv.add(
                self.__hough_lines
                , self.ch_masked
            ), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1
        )

    # Properties
    @property
    def bounding_box(self):
        return self.__bb

    @property
    def ch_masked(self):
        self.mask_with_convex_hull()
        return self.__ch_masked

    @property
    def recolored_image(self):
        return self.__recolored_image

    @property
    def test_image(self):
        # self.produce_test_image()
        self.produce_test_image()
        return self.__test

    @property
    def convex_hull(self):
        self.calculate_convex_hull()
        return self.__convex_hull

    @property
    def adaptive_threshold(self):
        if self.__adaptive_threshold is None:
            self.adaptive_thresholding(255, 15)
        return self.__adaptive_threshold

    @property
    def hough_lines(self):
        return self.__hough_lines

    @property
    def component_image(self):
        return self.__component_image

    @property
    def masked_image(self):
        return self.__masked_image

    @property
    def mask(self) -> np.ndarray:
        return self.__mask

    @mask.setter
    def mask(self, val):
        raise Exception("Call make_mask to create it!")

    @property
    def image_gray(self) -> np.ndarray:
        return self.__image_gray

    @image_gray.setter
    def image_gray(self, val):
        raise Exception("You can not modify the gray scale image!")

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @image.setter
    def image(self, val):
        raise Exception("The loaded image can not be modified! (call imread to change it)")

    @property
    def contour(self):
        return self.__contour