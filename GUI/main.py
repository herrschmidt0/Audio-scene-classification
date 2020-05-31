from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal

import sys
import os

from fileloadervisualizer import FileLoaderFeatureVisualizer
from clustering import Clustering
from classification import Classification

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

		self.body = QStackedWidget()

		#dimReductionAction = actionTools.addAction("Dimensionality reduction")
		#dimReductionAction.triggered.connect(self.showDimReduction)

		openFileAction = actionTools.addAction("Open file - Visualize features")
		openFileAction.triggered.connect(self.showFileLoader)
		
		fileLoaderFeatureVisualizer = FileLoaderFeatureVisualizer()
		fileLoaderFeatureVisualizer.audioLoadedSignal.connect(self.audioLoaded)
		self.body.addWidget(fileLoaderFeatureVisualizer)

		clusteringAction = actionTools.addAction("Clustering")
		clusteringAction.triggered.connect(self.showClustering)
		
		clusteringWidget = Clustering()
		
		scrollArea = QScrollArea()
		scrollArea.setWidgetResizable(True)
		scrollArea.setWidget(clusteringWidget)
		scrollArea.setFixedHeight(600)
		self.body.addWidget(scrollArea)

		classificationAction = actionTools.addAction("Classification")
		classificationAction.triggered.connect(self.showClassification)

		classificationWidget = Classification()

		scrollArea = QScrollArea()
		scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		scrollArea.setWidgetResizable(True)
		scrollArea.setWidget(classificationWidget)
		scrollArea.setFixedHeight(600)

		self.body.addWidget(scrollArea)


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

	def showClustering(self):

		self.body.setCurrentIndex(1)

	def showClassification(self):

		self.body.setCurrentIndex(2)



app = QApplication([])
win = MainWindow()
win.show()
app.exec_()