from src.GUI.App import App
import sys

if __name__ == '__main__':
    app = App(sys.argv)
    app.initUI()

    sys.exit(app.app.exec_())
# img = Image()
# img.imread("../assets/Rubik00.png")
#
# img.find_contours(30, 255)
# img.make_mask()
#
#
# img.debug()

