from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtMultimedia import *

import sys
import os

import numpy as np
import librosa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from minisom import MiniSom


class DimReduction(QWidget):

	def __init__(self, audio):
		super().__init__()
		self.audio = audio

		self.mainLayout = QVBoxLayout()
		self.setLayout(self.mainLayout)

		self.noiseReductionGroup = self.NoiseReductionGroup()
		self.mainLayout.addWidget(self.noiseReductionGroup)
	
		self.featureSelectorGroup = self.FeatureSelectorGroup()
		self.mainLayout.addWidget(self.featureSelectorGroup)
	
		self.methodSelectorGroup = self.MethodSelectorGroup()
		self.mainLayout.addWidget(self.methodSelectorGroup)
		self.methodSelectorGroup.startButton.clicked.connect(self.startProcess)

	def startProcess(self):

		noiseRedMethodId = self.noiseReductionGroup.methodSelector.checkedId()
		print(noiseRedMethodId)

		spectrumMethodId = self.featureSelectorGroup.spectrumSelector.checkedId()
		print(spectrumMethodId)

		includeExtraFeatures = self.featureSelectorGroup.featureCheckbox.isChecked()
		print(includeExtraFeatures)

		dimRedMethodId = self.methodSelectorGroup.methodSelector.currentIndex()
		print(dimRedMethodId)
		n_components = 10

		# Extract features based on selected method
		if spectrumMethodId == 0:
			
			stft = librosa.core.stft(self.audio['y'])
			stft_log = librosa.amplitude_to_db(abs(stft))
			input = stft_log

		elif spectrumMethodId == 1:
			mfcc = librosa.feature.mfcc(self.audio['y'], sr=self.audio['sr'], n_mfcc=50)
			input = mfcc
		else:
			print("Spectrum method ID is invalid:", spectrumMethodId)

		# Apply dimensionality reduction
		if dimRedMethodId == 0:
			
			pca = PCA(n_components=n_components)
			pca.fit(input)

			output = pca.transform(input)

		elif dimRedMethodId == 1:

			output_partial = input
			if len(input[0]) > 50:
				pca = PCA(n_components=50)
				output_partial = pca.fit_transform(input)

			tsne = TSNE(n_components=3)
			output = tsne.fit_transform(output_partial)

		elif dimRedMethodId == 2:
			
			isomap = Isomap(n_components=n_components)
			output = isomap.fit_transform(input)

		elif dimRedMethodId == 3:
			
			som = MiniSom(n_components, n_components, len(input[0]), sigma=0.3, learning_rate=0.5)
			output = som.train_batch(input, 100)

		else:
			print("Dim red method ID is invalid:", dimRedMethodId)

		
		data = {
			'input': input,
			'output': output
		}

		dataTableInspector = self.DataTableInspector(data)
		self.mainLayout.addWidget(dataTableInspector)


	class NoiseReductionGroup(QGroupBox):
		def __init__(self):
			super().__init__()

			self.setTitle("Noise Reduction - (optional)")
			layout = QHBoxLayout()
			self.setLayout(layout)

			radioButtonGroup = QButtonGroup()
			radio1 = QRadioButton("None")
			radio1.setChecked(True)
			radio2 = QRadioButton("Bandpass filter")
			radio3 = QRadioButton("Threshold filter")

			radioButtonGroup.addButton(radio1, 0)
			radioButtonGroup.addButton(radio2, 1)
			radioButtonGroup.addButton(radio3, 2)
			self.methodSelector = radioButtonGroup

			layout.addWidget(radio1)
			layout.addWidget(radio2)
			layout.addWidget(radio3)


	class FeatureSelectorGroup(QGroupBox):
		def __init__(self):
			super().__init__()	
			self.setTitle("Select Features")

			layout = QVBoxLayout()
			self.setLayout(layout)

			# Column 1 - Spectrograms
			spectrogramSelector = QComboBox()
			spectrogramSelector.addItem("STFT (Short Time Fourier Transform)")
			spectrogramSelector.addItem("Melspectrogram (Mel Scaled STFT)")
			spectrogramSelector.addItem("MFCC (Mel Frequency Cepstrum)")

			self.spectrogramSelector = spectrogramSelector
			layout.addWidget(self.spectrogramSelector)

			nrofbinsLabel = QLabel("Nr of frequency bins:")
			layout.addWidget(nrofbinsLabel)
			nrofbinsEdit = QLineEdit()
			nrofbinsEdit.setValidator(QIntValidator(0, 200));
			nrofbinsEdit.setFixedSize(100,20);
			layout.addWidget(nrofbinsEdit)

			# Column 2 - Extracted features
			self.featureCheckbox = QCheckBox("Add extracted features")
			self.featureCheckbox.setChecked(False)
			self.featureCheckbox.stateChanged.connect(self.featureCheckboxChanged)
			layout.addWidget(self.featureCheckbox)

			featureList = ['RMS','Zero-crossing rate','Spectral centroid', 'Spectral Rolloff', 'Spectral Flux']
			self.featureCheckboxList = QListWidget()

			sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
			sizePolicy.setHorizontalStretch(1)
			self.featureCheckboxList.setSizePolicy(sizePolicy)

			for name in featureList:
				item = QListWidgetItem()
				item.setText(name)
				item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
				item.setCheckState(Qt.Unchecked)
				
				self.featureCheckboxList.addItem(item)

			self.featureCheckboxList.setDisabled(True)
			layout.addWidget(self.featureCheckboxList)

		def featureCheckboxChanged(self, _):
			if self.featureCheckbox.isChecked():
				self.featureCheckboxList.setDisabled(False)
			else:
				self.featureCheckboxList.setDisabled(True)


	class MethodSelectorGroup(QGroupBox):
		def __init__(self):
			super().__init__()	

			self.setTitle("Select Method")

			layout = QVBoxLayout()
			self.setLayout(layout)

			selector = QComboBox()
			selector.addItem("PCA")
			selector.addItem("t-SNE")
			selector.addItem("Isomap")
			selector.addItem("Self organizing map")
			layout.addWidget(selector)
			self.methodSelector = selector

			nrofcompLabel = QLabel("Nr of components")
			layout.addWidget(nrofcompLabel)
			
			self.startButton = QPushButton("Start!")
			layout.addWidget(self.startButton)

	class DataTableInspector(QTabWidget):
		def __init__(self, data):
			super().__init__()

			inputTab = self.TableTab(data['input'])
			outputTab = self.TableTab(data['output'])

			self.addTab(inputTab, "Input")
			self.addTab(outputTab, "Output")

		class TableTab(QWidget):
			def __init__(self, tableData):
				super().__init__()

				layout = QVBoxLayout()
				self.setLayout(layout)

				table = QTableWidget()
				layout.addWidget(table)

				#print(tableData)
				table.setRowCount(len(tableData))
				table.setColumnCount(len(tableData[0]))
				for i, row in enumerate(tableData):
					for j, col in enumerate(row): 
						table.setItem(i, j, QTableWidgetItem(str(tableData[i, j])))