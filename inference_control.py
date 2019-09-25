import sys
from PyQt4 import QtGui, uic

class inference_control(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(inference_control, self).__init__(parent)
        self.ui = uic.loadUi("inference_control.ui")
        self.ui.setStyleSheet("background-color: white")
        #self.ui.frame_3.setStyleSheet("background-color: darkGray")
        #self.ui.label.setStyleSheet("color: white")
        #self.ui.model_icon.setStyleSheet("background-color: white; color: white")
        self.ui.show()
        print("running")
