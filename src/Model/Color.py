import numpy as np


class Color:
    def __init__(self, y, u, v):
        self.y = y
        self.u = u
        self.v = v


class DefinedColor:
    def __init__(self, y_min, y_max, u_min, u_max, v_min, v_max, name):
        self.y = [y_min, y_max]
        self.u = [u_min, u_max]
        self.v = [v_min, v_max]

        y_h = y_max-y_min/2
        u_h = u_max-u_min/2
        v_h = v_max-v_min/2

        self.center = [y_min+y_h, u_min+u_h, v_min+v_h]
        self.name = name

    def __is_y(self, y):
        return self.y[0] <= y <= self.y[1]

    def __is_u(self, u):
        return self.u[0] <= u <= self.u[1]

    def __is_v(self, v):
        return self.v[0] <= v <= self.v[1]

    def is_color_inside(self, color: Color):
        return self.__is_y(color.y) and self.__is_u(color.u) and self.__is_v(color.v)

    def distance(self, color: Color):
        return np.sqrt((color.y-self.center[0])**2 + (color.u-self.center[1])**2 + (color.v-self.center[2]))


class ColorComponents:
    COLOR_COMPONENTS = [
        DefinedColor(50, 140, 72, 148, 160, 150, 'red'),
        DefinedColor(140, 210, 52, 148, 160, 150, 'orange'),
        DefinedColor(118, 150, 65, 115, 120, 160, 'yellow'),
        DefinedColor(70, 240, 140, 250, 0, 110, 'blue'),
        DefinedColor(70, 240, 5, 135, 10, 110, 'green'),
        DefinedColor(190, 255, 116, 130, 117, 139, 'white'),
    ]

    @staticmethod
    def get_components():
        return ColorComponents.COLOR_COMPONENTS

    @staticmethod
    def get_color(color: Color) -> str:
        dfc: DefinedColor
        for dfc in ColorComponents.COLOR_COMPONENTS:
            if dfc.is_color_inside(color):
                return dfc.name
        dst = np.inf
        clr = None
        for dfc in ColorComponents.COLOR_COMPONENTS:
            d = dfc.distance(color)
            if d < dst:
                dst = d
                clr = dfc
        return clr.name

