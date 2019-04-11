from src.GUI.App import App
import sys

if __name__ == '__main__':
    app = App(sys.argv)
    app.initUI()

    sys.exit(app.app.exec_())

