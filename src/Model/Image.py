import cv2 as cv
import numpy as np
import copy
from PyQt5.QtGui import QImage
import asyncio


class Image:
    def __init__(self):
        self.__image = None
        self.__image_gray = None
        self.__contour = None
        self.__mask = None
        self.__masked_image = None
        self.__component_image = None
        self.__canny = None
        self.__adaptive_threshold = None
        self.__convex_hull = None
        self.__test = None

    def imread(self, path: str):
        self.__image = cv.imread(path)
        self.__image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def find_contours(self, thresh, max_val, thresh_type=0, kernel=6, iterations=3):
        if self.image_gray is None:
            raise Exception("Read in an image with imread!")
        ret, thresh = cv.threshold(self.image_gray, thresh, max_val, thresh_type)
        kernel_matrix = np.ones((kernel, kernel), np.uint8)
        thresh = cv.erode(thresh, kernel_matrix, iterations=iterations)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img = copy.deepcopy(self.image)
        cv.drawContours(img, contours, -1, (0, 255, 0), 3)
        self.__contour = (img, contours, hierarchy)

    def make_mask(self):
        if self.contour is None:
            raise Exception("First generate the contours with find_contours!")
        img = np.ndarray(self.image.shape, np.uint8)
        cv.drawContours(img, self.contour[1], -1, (255, 255, 255), -1)
        self.__mask = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    def _mask_image(self):
        img = self.image.copy()
        b, g, r = cv.split(img)
        r = cv.bitwise_and(r, self.mask)
        g = cv.bitwise_and(g, self.mask)
        b = cv.bitwise_and(b, self.mask)
        self.__masked_image = cv.merge((b, g, r))

    def _components(self):
        retVal, labels = cv.connectedComponents(self.mask, None, 8, cv.CV_16U)
        labelsNorm = cv.normalize(labels, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)
        cv.imwrite("temp/components.png", labelsNorm)
        img = cv.imread("temp/components.png")
        self.__component_image = img

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

        drawing = np.zeros((self.adaptive_threshold.shape[0], self.adaptive_threshold.shape[1], 3), dtype=np.uint8)
        import random as rng
        for i in range(len(convex_hull)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            # cv.drawContours(drawing, contours, i, color)
            cv.drawContours(drawing, convex_hull, i, color, thickness=10)

        self.__convex_hull = drawing

    @classmethod
    def _distance(cls, coord1: list, coord2: list) -> float:
        x = coord1[0] - coord2[0]
        y = coord1[1] - coord2[1]
        return (x**2 - y**2)**(1/2)

    def grab_cut(self):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (0, 0, self.image.shape[0]-1, self.image.shape[1]-1)
        cv.grabCut(self.image, self.adaptive_threshold, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((self.adaptive_threshold == 2) | (self.adaptive_threshold == 0), 0, 1).astype('uint8')
        img = self.image * mask2[:, :, np.newaxis]
        self.__test = img

    def debug(self):
        cv.imshow("Colored Image", self.image)
        cv.imshow("Adaptive Threshold Image", self.adaptive_threshold)
        self.convex_hull()
        cv.imshow("Convex Hull", self.__convex_hull)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_canny(self, thresh_min=30, thresh_max=255):
        canny = cv.Canny(self.image, thresh_min, thresh_max)
        self.__canny = canny

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
                cv.cvtColor(self.adaptive_threshold, cv.COLOR_GRAY2BGR)
                , self.convex_hull
            ), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1
        )

    # Properties
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
    def canny(self):
        return self.__canny

    @property
    def component_image(self):
        if self.mask is None:
            self.make_mask()
            self._mask_image()
        self._components()
        return self.__component_image

    @property
    def masked_image(self):
        self._mask_image()
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
    def contour(self) -> tuple:
        return self.__contour

    @contour.setter
    def contour(self, val):
        raise Exception("Call find_contours to set it!")