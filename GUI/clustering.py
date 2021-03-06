from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtMultimedia import *

import math
from itertools import product
from zipfile import ZipFile
import json
import pprint

import numpy as np
import pandas as pd
import scipy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from minisom import MiniSom

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

n_classes = 5
colors = np.array(['blue', 'red', 'green', 'black', 'yellow'])


class Clustering(QWidget):

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
		groupLayout.addWidget(label, 2, 0)
		nrBins = QComboBox()
		nrBins.addItem("13")
		nrBins.addItem("20")
		nrBins.addItem("40")
		nrBins.setCurrentIndex(2)
		groupLayout.addWidget(nrBins, 3, 0)

		# Reshaping
		label = QLabel('Reshaping method')
		groupLayout.addWidget(label, 2, 1)
		reshapeMethod = QComboBox()
		reshapeMethod.addItem("1")
		reshapeMethod.addItem("2")
		reshapeMethod.addItem("3")
		reshapeMethod.setCurrentIndex(2)
		groupLayout.addWidget(reshapeMethod, 3, 1)

		# Dimensionality reduction - target: 2D
		label = QLabel('Dimensionality reduction')
		groupLayout.addWidget(label, 4, 0)
		drSelector = QComboBox()
		drSelector.addItem("PCA")
		drSelector.addItem("t-SNE")
		drSelector.addItem("Isomap")
		drSelector.addItem("Self organizing map")
		groupLayout.addWidget(drSelector, 5, 0)

		# Dimensionality reduction parameter
		label = QLabel('Dimensionality reduction Hyperparameter')
		groupLayout.addWidget(label, 4, 1)
		drHyperParam = QLineEdit()
		groupLayout.addWidget(drHyperParam, 5, 1)

		# Start button
		def collectParams():
			filterParam = str(filterSelector.currentIndex())
			feParam = ['stft', 'mel', 'mfcc'][feSelector.currentIndex()]
			binsParam = ['13', '20', '40'][nrBins.currentIndex()]
			reshapeParam = str(reshapeMethod.currentIndex() + 1)
			drParam = ['pca', 'tsne', 'isomap', 'som'][drSelector.currentIndex()]
			drParamParam = int(drHyperParam.text()) if drHyperParam.text().isnumeric() else 50

			params = [filterParam, feParam, binsParam, reshapeParam, drParam, drParamParam]

			results, labels, evaluation = self.applyClustering(params)

			self.displayResults(results, labels, evaluation)

		start = QPushButton()
		start.setText('Start!')
		start.clicked.connect(collectParams)
		groupLayout.addWidget(start, 6, 0)

		# Add param group to mainlayout
		paramGroup.setFixedHeight(300)
		self.mainLayout.addWidget(paramGroup)

		# Create canvas
		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.canvas.setFixedHeight(600)	
		self.canvas.setFixedWidth(600)
		self.mainLayout.addWidget(self.canvas)

		# Evaluation text
		self.eval_tedit = QPlainTextEdit()
		self.eval_tedit.setFixedHeight(300)	
		self.mainLayout.addWidget(self.eval_tedit)


	def applyClustering(self, params):

		def load_samples(fname_data, fname_labels):
			url = '../../Preprocessing/'

			# Load files
			spects = np.load(url+fname_data)
			labels = np.load(url+fname_labels)
			labels = np.argmax(labels, axis=1)

			# Distribution of labels
			print("Label distribution:", np.histogram(labels, bins=len(np.unique(labels))),'\nTotal:', len(labels))
			print("Nr of data samples:", len(spects))

			# Sample k elements from each class
			k = 150
			spects_sampled = []
			labels_sampled = []
			np.random.seed(17)

			for label in np.unique(labels)[:5]:
				idxs = np.squeeze(np.argwhere(labels == label)) 
				idxs = np.random.choice(idxs, size=k)

				spects_sampled.append(spects[idxs])
				labels_sampled.append([label for i in range(k)])

			spects_sampled = np.concatenate(spects_sampled, axis=0)
			labels_sampled = np.concatenate(labels_sampled, axis=0)

			# Shuffle
			p = np.random.permutation(len(labels_sampled))
			labels_sampled, spects_sampled = labels_sampled[p], spects_sampled[p]

			return labels_sampled, spects_sampled


		def cluster_eval(labels, features, n_classes):

			# Clustering models
			clusterers = [
				KMeans(init='k-means++', n_clusters=n_classes, n_init=10),
				AgglomerativeClustering(n_clusters=n_classes, linkage='ward'),
				SpectralClustering(n_clusters=n_classes),
				OPTICS(min_samples=2),
			]

			model = clusterers[0]
			model.fit(features)

			# Clustering metrics for the high dimensional data
			ms = [
				metrics.adjusted_rand_score(labels, model.labels_),
				metrics.normalized_mutual_info_score(labels, model.labels_),
				metrics.fowlkes_mallows_score(labels, model.labels_),
				metrics.v_measure_score(labels, model.labels_),
				metrics.homogeneity_score(labels, model.labels_),
				metrics.completeness_score(labels, model.labels_)
			]

			return ms

		# Helper
		def f(lowdata, highdata, labels, params):
		    # Print parameters
		    p = params
		    print(' '.join(map(str, p)))

		    # Normalize 2D output
		    scaler = StandardScaler()
		    scaler.fit(lowdata)
		    lowdata = scaler.transform(lowdata)

		    # Evaluation metrics
		    ms_high = cluster_eval(labels, highdata, n_classes)
		    ms_low = cluster_eval(labels, lowdata, n_classes)
		    evaluation = {
		        'params': params,
		        'high': ms_high,
		        'low': ms_low
		    }

		    # Append results
		    results = {
		        'data': lowdata,
		        'params': [{'name': 'p', 'value': val} for val in params]
		    }
		    return results, evaluation

		# Load data
		p = params
		fname_data = 'urbansound_'+'filter'+str(p[0])+'_'+p[1]+str(p[2])+'_reshape'+str(p[3])+'.npy'
		fname_labels = 'labels_urbansound_5.npy'
		labels, highdata = load_samples(fname_data, fname_labels)

		# Normalize
		scaler = StandardScaler()
		highdata = scaler.fit_transform(highdata)

		# Apply dimensionality reduction  
		results = None
		evaluation = None
		if p[4] == 'pca':
			for whitening in [False]:
				lowdata = PCA(n_components=2, whiten=whitening).fit_transform(highdata)
				results, evaluation = f(lowdata, highdata, 
				                      labels, p)

		elif p[4] == 'tsne':
			lowdata = TSNE(n_components=2,
			          perplexity=p[5],
			          n_iter=2500).fit_transform(highdata) 
			results, evaluation = f(lowdata, highdata, 
			                      labels, p)

		elif p[4] == 'isomap':
			lowdata = Isomap(n_components=2,
			          n_neighbors=p[5]).fit_transform(highdata)
			results, evaluation = f(lowdata, highdata, 
			                      labels, p)

		elif p[4] == 'som':
			input_len = highdata.shape[1]
			x = y = int(2*np.sqrt(highdata.shape[0]))
			som = MiniSom(x, y, 
			              input_len=input_len, sigma=p[5], learning_rate=0.5, 
			              neighborhood_function='gaussian')
			som.pca_weights_init(highdata)
			som.train_batch(highdata, 2000, verbose=True)  

			lowdata = []
			for i, hp in enumerate(highdata):
				w = som.winner(hp)  
				lowdata.append(w)
			results, evaluation = f(lowdata, highdata, 
			                        labels, p)

		return results, labels, evaluation


	def displayResults(self, results, labels, evaluation):
  
		# Removing outliers
		def remove_outliers(data):
			mean = np.mean(data, axis=0)
			dists = [np.linalg.norm(x-mean) for x in data]
			dists = sorted(dists)

			Q3 = np.quantile(dists, 0.9)

			idxs = [i for i, x in enumerate(data) if np.linalg.norm(x-mean) < Q3]
			return idxs

		# Calculate inliers
		inlier_idxs = remove_outliers(results['data'])
		inliers = results['data'][inlier_idxs]
		inlier_labels = labels[inlier_idxs]

		# Plot
		self.figure.clf()
		ax = self.figure.add_subplot(1, 1, 1)
		ax.scatter(inliers[:,0], 
		            inliers[:,1], 
		            s=3, c=colors[inlier_labels % len(colors)])
		s = [(x['name'] + ':' + str(x['value']) + ' ') for x in results['params']]
		ax.set_title(' '.join(s))

		self.canvas.draw()

		# Display evaluation metrics
		text = 'Adjusted rand score: ' + str(round(evaluation['low'][0], 2)) +\
			'\nNormalized mutual info score: ' + str(round(evaluation['low'][1], 2)) +\
			'\nFowlkes Mallows score:'  + str(round(evaluation['low'][2], 2)) +\
			'\nV_measure score: ' + str(round(evaluation['low'][3], 2)) +\
			'\nHomogeneity score: ' + str(round(evaluation['low'][4], 2)) +\
			'\nCompleteness score: ' + str(round(evaluation['low'][5], 2))

		self.eval_tedit.setPlainText(text)