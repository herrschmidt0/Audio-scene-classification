from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtMultimedia import *

import time
import math 
from itertools import product
import json
import pprint

import numpy as np
import pandas as pd
import scipy 

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from minisom import MiniSom

import librosa

from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sn

class_names = ['car_horn', 'dog_bark', 'street_music', 'gun_shot', 'engine_idling']
n_classes = 5

class Classification(QWidget):

	def __init__(self):
		super().__init__()

		self.mainLayout = QVBoxLayout()
		self.setLayout(self.mainLayout)

		# Hyperparameter selection
		paramGroup = QGroupBox()
		groupLayout = QGridLayout()
		paramGroup.setLayout(groupLayout)

		# Filtering
		label = QLabel('Filter type')
		groupLayout.addWidget(label, 0, 0)
		filterSelector = QComboBox()
		filterSelector.addItem("No filtering")
		filterSelector.addItem("Bandpass 1, 1-10000")
		filterSelector.addItem("Bandpass 2, 25-9000")
		filterSelector.addItem("Bandpass 3, 50-8500")
		filterSelector.addItem("Bandpass 4, 100-8000")
		filterSelector.addItem("Bandpass 5, 150-7000")
		groupLayout.addWidget(filterSelector, 1, 0)


		# Feature extraction
		label = QLabel('Feature extraction')
		groupLayout.addWidget(label, 0, 1)
		feSelector = QComboBox()
		feSelector.addItem("STFT (Short Time Fourier Transform)")
		feSelector.addItem("Melspectrogram (Mel Scaled STFT)")
		feSelector.addItem("MFCC (Mel Frequency Cepstrum)")
		feSelector.setCurrentIndex(2)
		groupLayout.addWidget(feSelector, 1, 1)


		# Nr of frequency bins
		label = QLabel('Nr of frequency bins')
		groupLayout.addWidget(label, 0, 2)
		nrBins = QComboBox()
		nrBins.addItem("13")
		nrBins.addItem("20")
		nrBins.addItem("40")
		nrBins.setCurrentIndex(2)
		groupLayout.addWidget(nrBins, 1, 2)

		# Reshaping
		label = QLabel('Reshaping method')
		groupLayout.addWidget(label, 2, 0)
		reshapeMethod = QComboBox()
		reshapeMethod.addItem("1")
		reshapeMethod.addItem("2")
		reshapeMethod.addItem("3")
		reshapeMethod.setCurrentIndex(2)
		groupLayout.addWidget(reshapeMethod, 3, 0)

		# Dimensionality reduction
		label = QLabel('Dimensionality reduction')
		groupLayout.addWidget(label, 2, 1)
		drSelector = QComboBox()
		drSelector.addItem("PCA")
		drSelector.addItem("t-SNE")
		drSelector.addItem("Isomap")
		drSelector.addItem("Self organizing map")
		groupLayout.addWidget(drSelector, 3, 1)

		# Dimensionality reduction target
		label = QLabel('Dimensionality reduction target dimensionality')
		groupLayout.addWidget(label, 4, 0)
		drTarget = QLineEdit()
		groupLayout.addWidget(drTarget, 5, 0)

		# Dimensionality reduction parameter
		label = QLabel('Dimensionality reduction Hyperparameter')
		groupLayout.addWidget(label, 4, 1)
		drHyperParam = QLineEdit()
		groupLayout.addWidget(drHyperParam, 5, 1)


		# Start button - start classification process, then display results
		start = QPushButton()
		start.setText('Start!')
		def collectParams():
			filterParam = str(filterSelector.currentIndex())
			feParam = ['stft', 'mel', 'mfcc'][feSelector.currentIndex()]
			binsParam = ['13', '20', '40'][nrBins.currentIndex()]
			reshapeParam = str(nrBins.currentIndex() + 1)
			drParam = ['pca', 'tsne', 'isomap', 'som'][drSelector.currentIndex()]
			drTargetParam = int(drTarget.text())
			drParamParam = int(drHyperParam.text()) if drHyperParam.text().isnumeric() else 0

			params = [filterParam, feParam, binsParam, reshapeParam, drParam, drTargetParam, drParamParam]

			results = self.startClassification(params)
			self.displayResults(results)

		start.clicked.connect(collectParams)
		groupLayout.addWidget(start, 6, 0)

		paramGroup.setFixedHeight(300)
		self.mainLayout.addWidget(paramGroup)


	def startClassification(self, params):

		def load_samples(fname_data, fname_labels, class_size):
			url = '../../Preprocessing/'

			# Load files
			spects = np.load(url+fname_data)
			labels = np.load(url+fname_labels)
			labels = np.argmax(labels, axis=1)

			# Distribution of labels
			print("Label distribution:", np.histogram(labels, bins=len(np.unique(labels))),'\nTotal:', len(labels))
			print("Nr of data samples:", len(spects))

			# Sample k elements from each class
			spects_sampled = []
			labels_sampled = []
			np.random.seed(17)
			unique_labels = np.unique(labels)

			for label in unique_labels:
				idxs = np.squeeze(np.argwhere(labels == label))
				# Is k bigger than class size?
				kk=class_size
				if kk > len(idxs):
					kk =  len(idxs)
				idxs = np.random.choice(idxs, size=kk)

				spects_sampled.append(spects[idxs])
				# One-hot encode
				ohe = np.zeros(shape=(len(unique_labels)))
				ohe[label] = 1
				labels_sampled.append([ohe for i in range(kk)])

			spects_sampled = np.concatenate(spects_sampled, axis=0)
			labels_sampled = np.concatenate(labels_sampled, axis=0)

			# Shuffle
			p = np.random.permutation(len(labels_sampled))
			labels_sampled, spects_sampled = labels_sampled[p], spects_sampled[p]

			return labels_sampled, spects_sampled

		def eval_metrics(model, X_test, y_test, class_names):

			y_pred = model.predict(X_test)
			y_test = np.argmax(y_test, axis=1)
			y_pred = np.argmax(y_pred, axis=1)

			report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
			conf_m = confusion_matrix(y_test, y_pred)
			report['confusion_matrix'] = conf_m.tolist()
			return report

		def train_model(lowdata, labels, results, p):

			# Print parameters
			print('\n'+' '.join(map(str, p)))

			# Split data
			#X_train, X_test, y_train, y_test = train_test_split(lowdata, labels,  test_size=0.2, random_state=29, shuffle=True)

			# Cross validation
			n_split = 5
			scores = []
			for train_index, test_index in KFold(n_split).split(lowdata):

				X_train, X_test = lowdata[train_index], lowdata[test_index]
				y_train, y_test = labels[train_index], labels[test_index]

				# Define model (complexity is a function of input dimensionality)
				if p[5] > 8:
				  k = 32
				elif p[5] >= 3:
				  k = 8
				else:
				  k = 3

				model=Sequential()
				model.add(Dense(2*k, activation='relu', input_dim=X_train.shape[1]))
				model.add(Dropout(0.5))
				model.add(Dense(k, activation='relu'))
				model.add(Dense(n_classes, activation='softmax'))

				loss = categorical_crossentropy
				#optimizer = Adadelta(lr=0.0005)
				optimizer = Adam(lr=0.0005)
				model.compile(loss=loss,
				            optimizer=optimizer,
				            metrics=['categorical_accuracy'])
				#print(model.summary())

				# Train model
				n_epochs = 30
				history = model.fit(X_train, y_train,
				        batch_size=32,
				        epochs=n_epochs,
				        verbose=0,
				        validation_data=(X_test, y_test))

				#plot(history)
				scores.append(eval_metrics(model, X_test, y_test, class_names))

			# Evaluate model
			results = {
			  'params': p,
			  'history': history.history,
			  'score': scores
			}
			return results 

		# Grid search
		p=params
		results = None
		evaluation = None

		# Load data
		fname_data = 'urbansound_'+'filter'+str(p[0])+'_'+p[1]+str(p[2])+'_reshape'+str(p[3])+'.npy'
		fname_labels = 'labels_urbansound_5.npy'
		labels, highdata = load_samples(fname_data, fname_labels, class_size=500)

		# Normalize
		scaler = StandardScaler()
		highdata = scaler.fit_transform(highdata)

		# Apply dimensionality reduction  
		if p[4] == 'pca':
			lowdata = PCA(n_components=p[5], whiten=False).fit_transform(highdata)
			results = train_model(lowdata, labels, results, p)

		elif p[4] == 'tsne':
			lowdata = TSNE(n_components=p[5],
			          perplexity=p[6],
			          n_iter=2500, method='exact').fit_transform(highdata) 
			results = train_model(lowdata, labels, results, p)

		elif p[4] == 'isomap':
			lowdata = Isomap(n_components=p[5],
			          n_neighbors=p[6]).fit_transform(highdata)
			results = train_model(lowdata, labels, results, p)

		elif p[4] == 'som':
			pass

		return results

	def displayResults(self, results):

		# Display overall accuracy
		text = 'Overall accuracy: ' + str(round(np.mean([x['accuracy'] for x in results['score']]), 3))
		acc_tedit = QLineEdit(text)
		self.mainLayout.addWidget(acc_tedit)

		# Plot learning curve
		figure = Figure()
		self.canvas = FigureCanvas(figure)
		self.canvas.setFixedHeight(600)	
		self.mainLayout.addWidget(self.canvas)

		ax1 = figure.add_subplot(2,2,1) 

		# Plot training & validation accuracy values
		ax1.plot(results['history']['categorical_accuracy'])
		ax1.plot(results['history']['val_categorical_accuracy'])
		ax1.set_title('Model accuracy')
		ax1.set_ylabel('Accuracy')
		ax1.set_xlabel('Epoch')
		ax1.legend(['Train', 'Test'], loc='upper left')

		ax2 = figure.add_subplot(2,2,2) 

		# Plot training & validation loss values
		ax2.plot(results['history']['loss'])
		ax2.plot(results['history']['val_loss'])
		ax2.set_title('Model loss')
		ax2.set_ylabel('Loss')
		ax2.set_xlabel('Epoch')
		ax2.legend(['Train', 'Test'], loc='upper left')

		ax3 = figure.add_subplot(2,2,3)
		sn.heatmap(results['score'][4]['confusion_matrix'], ax=ax3, annot=True)
		
		self.canvas.draw()

		# Display classification report
		pp = pprint.PrettyPrinter(indent=3)
		text = pp.pformat(results['score'])
		#text = json.dumps(results['score'], sort_keys=True, indent=4)
		
		report_tedit = QPlainTextEdit(text)
		report_tedit.setFixedHeight(600)	
		self.mainLayout.addWidget(report_tedit)