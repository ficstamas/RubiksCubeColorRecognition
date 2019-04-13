import cv2 as cv
import numpy as np
import copy
from PyQt5.QtGui import QImage
import asyncio
import math
from src.Model.Color import *


class Image:
    def __init__(self):
        self.__image = None
        self.__image_gray = None
        self.__corners = None
        self.__smoothed = None
        self.__result_image = None

    def find_corners(self):
        theta_error = 30
        canny = cv.Canny(self.__image_gray, 50, 200)
        lines = cv.HoughLinesP(canny, 1, np.pi/360, 50, None, 15, 20)

        # Sorting lines by length and calculating slopes and degrees
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

        # Sort lines by X coordinate
        ordered = sorted(parallel_filtered, key=lambda x: x[0])[:2]
        # Moving to end points and begin points
        for i in range(ordered.__len__()):
            if i == ordered.__len__()-1:
                continue
            l1 = ordered[i]
            l2 = ordered[i+1]
            delta = (l1[2]-l2[0], l1[3]-l2[1])
            l2[0] += delta[0]
            l2[1] += delta[1]
            l2[2] += delta[0]
            l2[3] += delta[1]

        # selecting the face
        pt1 = [ordered[0][0], ordered[0][1]]
        pt2 = [ordered[0][2], ordered[0][3]]
        pt3 = [ordered[1][2], ordered[1][3]]
        pt4 = [ordered[1][2]+ordered[0][0]-ordered[0][2], ordered[1][3]+ordered[0][1]-ordered[0][3]]

        sides = np.array([pt1, pt2, pt3, pt4], np.int32)
        sides = sides.reshape((-1, 1, 2))

        # Creating mask for the face
        mask = np.zeros(shape=self.image.shape, dtype=np.uint8)
        cv.fillPoly(mask, [sides], (255, 255, 255))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # Masking out the face
        masked_image = self.image.copy()
        b, g, r = cv.split(masked_image)

        b = cv.bitwise_and(b, mask)
        g = cv.bitwise_and(g, mask)
        r = cv.bitwise_and(r, mask)

        masked_image = cv.merge((b, g, r))
        cv.polylines(masked_image, [sides], True, (120, 30, 200), 3)

        # Calculating intersection lines
        p1 = sides[0][0]
        p2 = sides[1][0]
        p3 = sides[2][0]
        delta = [p2[0]-p1[0], p2[1]-p1[1]]
        delta2 = [p3[0]-p2[0], p3[1]-p2[1]]

        intersection_lines = []
        for i in range(1, 3):
            b1 = [p1[0]+int(delta[0]*i/3), p1[1]+int(delta[1]*i/3)]
            e1 = [b1[0]+delta2[0], b1[1]+delta2[1]]
            intersection_lines.append([b1, e1])
        for i in range(1, 3):
            b1 = [p2[0]+int(delta2[0]*i/3), p2[1]+int(delta2[1]*i/3)]
            e1 = [b1[0]-delta[0], b1[1]-delta[1]]
            intersection_lines.append([b1, e1])

        # Draw intersection lines
        for line in intersection_lines:
            b1 = line[0]
            e1 = line[1]
            cv.line(masked_image, tuple(b1), tuple(e1), (200, 30, 50), 3)

        def add_vectors(v1, v2):
            return [v1[0]+v2[0], v1[1]+v2[1]]

        def divide_vector(v1, scalar):
            return [int(v1[0]/scalar), int(v1[1]/scalar)]

        def multiply_vector(v1, scalar):
            return [int(v1[0]*scalar), int(v1[1]*scalar)]

        # Locate cells
        p1 = sides[0][0]
        p2 = sides[1][0]
        p3 = sides[2][0]
        delta = [p2[0] - p1[0], p2[1] - p1[1]]
        delta2 = [p3[0] - p2[0], p3[1] - p2[1]]
        delta = divide_vector(delta, 3)
        delta2 = divide_vector(delta2, 3)
        blocks = []
        for i in range(1, 4):
            for j in range(1, 4):
                s0 = add_vectors(add_vectors(p1, multiply_vector(delta, i-1)), multiply_vector(delta2, j-1))
                s1 = add_vectors(add_vectors(p1, multiply_vector(delta, i)), multiply_vector(delta2, j-1))
                s2 = add_vectors(add_vectors(p1, multiply_vector(delta, i-1)), multiply_vector(delta2, j))
                s3 = add_vectors(add_vectors(p1, multiply_vector(delta, i)), multiply_vector(delta2, j))
                blocks.append([s0, s1, s3, s2])

        # Applying color balance
        cb_image = Image.color_balance(masked_image, 2)

        # TODO Avarage area and detect Color with the class
        first_section = blocks[0]

        # import random
        # for block in blocks:
        #     r = random.randint(0, 255)
        #     g = random.randint(0, 255)
        #     b = random.randint(0, 255)
        #     cv.fillPoly(masked_image, np.array([block]), (b, g, r))
        #
        # # Drawing
        # drawing = np.zeros(shape=self.image.shape, dtype=np.uint8)
        #
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         l = lines[i][0]
        #         cv.line(drawing, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 3, cv.LINE_AA)
        #
        # if best_lines is not None:
        #     for i in range(0, len(best_lines)):
        #         l = best_lines[i]
        #         cv.line(drawing, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 3, cv.LINE_AA)
        #
        # if parallel_filtered is not None:
        #     for i in range(0, len(parallel_filtered)):
        #         l = parallel_filtered[i]
        #         cv.line(drawing, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 255, 255), 3, cv.LINE_AA)
        #
        # if ordered is not None:
        #     for i in range(0, len(ordered)):
        #         l = ordered[i]
        #         cv.line(drawing, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 3, cv.LINE_AA)
        #
        # cv.fillPoly(drawing, [sides], (120, 30, 200))

        self.__result_image = cb_image

    @staticmethod
    def color_balance(img, percent=30):
        assert img.shape[2] == 3
        assert 0 < percent < 100

        half_percent = percent / 200.0

        channels = cv.split(img)

        out_channels = []
        for channel in channels:
            assert len(channel.shape) == 2
            # find the low and high precentile values (based on the input percentile)
            height, width = channel.shape
            vec_size = width * height
            flat = channel.reshape(vec_size)

            assert len(flat.shape) == 1

            flat = np.sort(flat)

            n_cols = flat.shape[0]

            low_val = flat[math.floor(n_cols * half_percent)]
            high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

            # saturate below the low percentile and above the high percentile
            thresholded = Image.apply_threshold(channel, low_val, high_val)
            # scale the channel
            normalized = cv.normalize(thresholded, thresholded.copy(), 0, 255, cv.NORM_MINMAX)
            out_channels.append(normalized)
        return cv.merge(out_channels)

    @staticmethod
    def apply_mask(matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()

    @staticmethod
    def apply_threshold(matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = Image.apply_mask(matrix, low_mask, low_value)

        high_mask = matrix > high_value
        matrix = Image.apply_mask(matrix, high_mask, high_value)

        return matrix

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
