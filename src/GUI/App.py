import sys
from PyQt5.QtWidgets import QApplication, QWidget
from src.GUI.MainWindow import Window


class App:
    def __init__(self, args):
        self.__app = QApplication(args)
        self.__app.setStyle('Fusion')

        self.__window = None

    def initUI(self):
        self.__window = Window()

    @property
    def app(self):
        return self.__app

    @app.setter
    def app(self, val):
        raise Exception("You can not change the main app!")
