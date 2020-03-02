from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtMultimedia import *

import sys
import os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np
import librosa
import librosa.display

class FileLoaderFeatureVisualizer(QWidget):

	audioLoadedSignal = pyqtSignal(object)

	def __init__(self):
		super().__init__()

		mainLayout = QVBoxLayout()
		self.setLayout(mainLayout)

		fileDetailsGroup = self.FileDetails()
		mainLayout.addWidget(fileDetailsGroup)
		fileDetailsGroup.audioLoadedSignal.connect(self.audioLoaded)

		spectrogramGroup = self.Spectrograms()
		mainLayout.addWidget(spectrogramGroup)
		fileDetailsGroup.audioLoadedSignal.connect(spectrogramGroup.mfccTab.plotSpectograms)
		fileDetailsGroup.audioLoadedSignal.connect(spectrogramGroup.stftTab.plotSpectograms)

		featureGroup = self.FeatureGroup()
		fileDetailsGroup.audioLoadedSignal.connect(featureGroup.plotFeatures)
		mainLayout.addWidget(featureGroup)

	def audioLoaded(self, audio):
		self.audioLoadedSignal.emit(audio)


	# FILE DETAILS PANEL
	class FileDetails(QGroupBox):

		audioLoadedSignal = pyqtSignal(object)

		def __init__(self):
			super().__init__()

			self.setTitle("File upload")
			self.layout = QGridLayout()
			self.setLayout(self.layout)

			self.uploadButton = QPushButton("File upload")
			self.uploadButton.clicked.connect(self.openFileDialog)
			self.layout.addWidget(self.uploadButton, 0, 0)
			self.fileName = QLabel("File name:")
			self.layout.addWidget(self.fileName, 0, 1)

			playerLayout = QHBoxLayout()
			self.layout.addLayout(playerLayout, 1, 0)
			playButton = QToolButton()
			playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
			playButton.clicked.connect(self.playAudio)
			playerLayout.addWidget(playButton)
			pauseButton = QToolButton()
			pauseButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
			playerLayout.addWidget(pauseButton)
			stopButton = QToolButton()
			stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
			stopButton.clicked.connect(self.stopAudio)
			playerLayout.addWidget(stopButton)

			self.formatLabel = QLabel("Format:")
			self.layout.addWidget(self.formatLabel, 0, 2)
			self.durationLabel = QLabel("Duration:")
			self.layout.addWidget(self.durationLabel, 1, 2)
			self.samplingRateLabel = QLabel("Sampling rate:")
			self.layout.addWidget(self.samplingRateLabel, 2, 2)
			self.bitDepthLabel = QLabel("Bit depth:")
			self.layout.addWidget(self.bitDepthLabel, 3, 2)


		def openFileDialog(self):
			fpath = QFileDialog.getOpenFileName(self, 'Open file', '\\',"Audio files (*.wav *.flac)")
			#print(fname)
			
			if os.path.exists(fpath[0]):
				self.fileName.setText(fpath[0].split('/')[-1])

				# Load audio file into Librosa
				self.y, self.sr = librosa.load(fpath[0])
				self.audioData = {
				'y': self.y,
				'sr': self.sr
				}
				self.audioLoadedSignal.emit(self.audioData)

				# Load audio file into qt sound player
				self.qtaudio = QSound(fpath[0])

				self.updateFileDetails()

		def updateFileDetails(self):
			self.samplingRateLabel.setText("Sampling rate: " + str(self.sr) + " Hz")

			duration = librosa.core.get_duration(y=self.y, sr=self.sr)
			self.durationLabel.setText("Duration: " + str(duration) + " seconds")

		def playAudio(self):
			if self.qtaudio != None:
				self.qtaudio.play()

		def stopAudio(self):
			if self.qtaudio != None:
				self.qtaudio.stop()
	

	# SPECTROGRAM TAB GROUP
	class Spectrograms(QTabWidget):

		def __init__(self):
			super().__init__()

			self.stftTab = self.StftTab()
			self.mfccTab = self.MfccTab()

			self.addTab(self.stftTab, "STFT")
			self.addTab(self.mfccTab, "MFCC")

		class StftTab(QWidget):
			def __init__(self):
				super().__init__()

				layout = QVBoxLayout()
				self.setLayout(layout)

				self.figure = Figure()
				self.canvas = FigureCanvas(self.figure)			
				layout.addWidget(self.canvas)

			def plotSpectograms(self, audio):

				stft = librosa.core.stft(audio['y'])
				S = librosa.amplitude_to_db(abs(stft))
				
				ax = self.figure.add_subplot(111)   		
				librosa.display.specshow(S, x_axis='time', ax=ax)
				ax.plot()

				self.canvas.draw()

		class MfccTab(QWidget):
			def __init__(self):
				super().__init__()
				
				layout = QVBoxLayout()
				self.setLayout(layout)

				self.figure = Figure()
				self.canvas = FigureCanvas(self.figure)			
				layout.addWidget(self.canvas)

			def plotSpectograms(self, audio):

				mfcc = librosa.feature.mfcc(audio['y'], sr=audio['sr'], n_mfcc=50)
				#print(mfcc)
				
				ax = self.figure.add_subplot(111)   		
				librosa.display.specshow(mfcc, x_axis='time', ax=ax)
				ax.plot()

				self.canvas.draw()


	## FEATURE TAB GROUP
	class FeatureGroup(QTabWidget):

		def __init__(self):
			super().__init__()

			tabTitles = ['Spectral centroids', 'Spectral bandwidths']
			self.tabs = []

			for title in tabTitles:
				tab = self.DiagramTab()
				self.tabs.append(tab) 
				self.addTab(tab, title)

		def plotFeatures(self, audio):

			centroids = librosa.feature.spectral_centroid(y=audio['y'], sr=audio['sr'])
			bandwidths = librosa.feature.spectral_bandwidth(y=audio['y'], sr=audio['sr'])

			data = [centroids, bandwidths]

			for i, tab in enumerate(self.tabs):
				tab.draw(data[i])

		class DiagramTab(QWidget):
			def __init__(self):
				super().__init__()

				layout = QVBoxLayout()
				self.setLayout(layout)

				self.figure = Figure()
				self.canvas = FigureCanvas(self.figure)			
				layout.addWidget(self.canvas)

			def draw(self, data):

				ax = self.figure.add_subplot(111)   		
				librosa.display.specshow(data, x_axis='time', ax=ax)
				ax.plot()
				self.canvas.draw()