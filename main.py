from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal

import sys
import os

from gui.fileloadervisualizer import FileLoaderFeatureVisualizer
from gui.dimreduction import DimReduction


class MainWindow(QMainWindow):

	def __init__(self):
		super().__init__()

		self.setWindowTitle("Comparison of dimensionality reduction methods")	
		#win.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowType_Mask)
		self.setGeometry(50, 50, 1200, 800)

		# Central widget
		centralw = QWidget()
		self.setCentralWidget(centralw)

		mainLayout = QVBoxLayout()
		centralw.setLayout(mainLayout)

		# Create menubar
		menubar = QMenuBar()
		mainLayout.addWidget(menubar)

		actionFile = menubar.addMenu("File")
		quitAction = actionFile.addAction("Quit")
		quitAction.triggered.connect(self.closeApplication)

		actionTools = menubar.addMenu("Tools")
		openFileAction = actionTools.addAction("Open file - Visualize features")

		dimReductionAction = actionTools.addAction("Dimensionality reduction")
		dimReductionAction.triggered.connect(self.showDimReduction)

		# Create QStackedWidget object and set the "File Loader & Feature Visualizer" view as the current one
		fileLoaderFeatureVisualizer = FileLoaderFeatureVisualizer()
		fileLoaderFeatureVisualizer.audioLoadedSignal.connect(self.audioLoaded)

		self.body = QStackedWidget()
		self.body.addWidget(fileLoaderFeatureVisualizer)
		openFileAction.triggered.connect(self.showFileLoader)
		mainLayout.addWidget(self.body)

		# There is no file loaded yet
		self.isAudioLoaded = False
	
	def audioLoaded(self, audio):
		self.isAudioLoaded = True
		self.audio = audio

	def closeApplication(self):
		sys.exit()

	def showFileLoader(self):
		self.body.setCurrentIndex(0)

	def showDimReduction(self):
		if self.isAudioLoaded == True:
			dimReduction = DimReduction(self.audio)
			self.body.addWidget(dimReduction)
			self.body.setCurrentIndex(1)



app = QApplication([])
win = MainWindow()
win.show()
app.exec_()