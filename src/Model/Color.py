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
        self.name = name

    def __is_y(self, y):
        return self.y[0] <= y <= self.y[1]

    def __is_u(self, u):
        return self.u[0] <= u <= self.u[1]

    def __is_v(self, v):
        return self.v[0] <= v <= self.v[1]

    def is_color_inside(self, color: Color):
        return self.__is_y(color.y) and self.__is_u(color.u) and self.__is_v(color.v)


class ColorComponents:
    COLOR_COMPONENTS = [
        DefinedColor(50, 140, 72, 148, 160, 150, 'red'),
        DefinedColor(140, 210, 52, 148, 160, 150, 'orange'),
        DefinedColor(70, 240, 140, 250, 0, 110, 'blue'),
        DefinedColor(70, 240, 5, 135, 10, 110, 'green'),
        DefinedColor(190, 255, 116, 130, 117, 139, 'white'),
        DefinedColor(118, 150, 65, 115, 120, 160, 'yellow'),
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
        return "None"

