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
        self.__corners = None
        self.__smoothed = None
        self.__result_image = None
        self.__color_pallet = {
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'blue': [0, 93, 167],
            'green': [0, 122, 30],
            'red': [206, 30, 33],
            'orange': [255, 96, 2],
            'yellow': [237, 190, 4]
        }

    def find_corners(self):
        theta_error = 30
        canny = cv.Canny(self.__image_gray, 50, 200)
        hough = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        lines = cv.HoughLinesP(canny, 1, np.pi/360, 50, None, 15, 20)

        sorted_lines = []
        for line in lines:
            l = list(line[0])
            dst = (l[0]-l[2])**2 + (l[1]-l[3])**2
            dst = np.sqrt(dst)
            slope = (l[3] - l[1]) / (l[2] - l[0])
            l.append(dst)
            l.append(slope)
            deg = np.rad2deg(math.atan(slope))+90

            l.append(deg)
            sorted_lines.append(l)
        sorted_lines = sorted(sorted_lines, key=lambda x: x[4], reverse=True)
        best_lines = sorted_lines[:12]

        # Filter out parallel lines
        add_item = True
        parallel_filtered = []
        for line in best_lines:
            if parallel_filtered.__len__() == 0:
                parallel_filtered.append(line)
                continue
            for sf in parallel_filtered:
                lower_bound = sf[-1]-theta_error
                upper_bound = sf[-1]+theta_error
                if lower_bound < 0:
                    lower_bound = 180+lower_bound
                if upper_bound > 180:
                    upper_bound = upper_bound-180
                if lower_bound <= upper_bound:
                    if lower_bound < line[-1] < upper_bound:
                        add_item = False
                        break
                else:
                    if 0 <= line[-1] < upper_bound or lower_bound < line[-1] <= 180:
                        add_item = False
                        break
            if add_item:
                parallel_filtered.append(line)
            add_item = True

        for line in parallel_filtered:
            

        # Drawing
        drawing = np.zeros(shape=self.image.shape, dtype=np.uint8)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv.line(drawing, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 3, cv.LINE_AA)

        if best_lines is not None:
            for i in range(0, len(best_lines)):
                l = best_lines[i]
                cv.line(drawing, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 3, cv.LINE_AA)

        if parallel_filtered is not None:
            for i in range(0, len(parallel_filtered)):
                l = parallel_filtered[i]
                cv.line(drawing, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 3, cv.LINE_AA)

        self.__result_image = drawing

    def imread(self, path: str):
        self.__image = cv.imread(path)
        self.__smoothed = cv.GaussianBlur(self.__image, (5, 5), 0)
        self.__image_gray = cv.cvtColor(self.__smoothed, cv.COLOR_BGR2GRAY)

    @classmethod
    def image_cv2qt(cls, img) -> QImage:
        imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, channel = imgrgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(imgrgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

        return qImg

    @property
    def image(self):
        return self.__image

    @property
    def result_image(self):
        self.find_corners()
        return self.__result_image
