
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from api.dim_reduction import DimReductor

import librosa

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2

#DIM_RED = 1
#n_classes = 2
method = 'mel'

results = open('results/results.txt', 'a')

# Grid search
for n_classes in range(2,4):
	for DIM_RED in range(0,2):

		results.write('Classes:' + str(n_classes) + '\nDimensionality reduction:'+str(DIM_RED)+'\n\n')

		spects = np.load('spectrograms_'+str(n_classes)+'class_'+method+'.npy')
		labels = np.load('labels_'+str(n_classes)+'class.npy')
		print("Spectrogram array shape:", spects.shape)

		# Dim reduction (spectral, temporal??)
		if DIM_RED:

			dimred_st = time.time()
			sp_dimred = []
			for spec in spects:

				#nr_new_dims = int(1*min(spec.shape[1], spec.shape[0]))
				nr_new_dims = 25

				pca = PCA(n_components=nr_new_dims)
				spec_new = pca.fit_transform(spec.T)
				#spec_new = pca.inverse_transform(spec_new)

				spec_new = spec_new.T

				#scaler = MinMaxScaler()
				#spec_new = scaler.fit_transform(spec_new)
				
				sp_dimred.append(spec_new)

			dimred_end = time.time()
			print("Duration of dim reduction:", round(dimred_end-dimred_st,2))

			spects = np.array(sp_dimred)

		spects = np.reshape(spects, (spects.shape[0], spects.shape[1], spects.shape[2], 1))
		print("Dim reduced array shape:", spects.shape)


		# Split data
		X_train, X_test, y_train, y_test = train_test_split(spects, labels, test_size=0.35, random_state=42)


		# Create CNN model 
		reg_term = 0.1
		ks = 3
		model = Sequential()

		model.add(Conv2D(32, kernel_size=(ks, ks), activation='elu', input_shape=X_train[0].shape, 
			kernel_regularizer=l2(reg_term), bias_regularizer=l2(reg_term)))
		model.add(BatchNormalization())

		model.add(Conv2D(32, kernel_size=(ks, ks), activation='elu',
			kernel_regularizer=l2(reg_term), bias_regularizer=l2(reg_term)))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(1, 2)))
		model.add(Dropout(0.3))

		model.add(Conv2D(64, kernel_size=(ks, ks), activation='elu', 
			kernel_regularizer=l2(reg_term), bias_regularizer=l2(reg_term)))
		model.add(BatchNormalization())
		
		model.add(Conv2D(64, kernel_size=(ks, ks), activation='elu', 
			kernel_regularizer=l2(reg_term), bias_regularizer=l2(reg_term)))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(1, 2)))
		model.add(Dropout(0.3))

		model.add(Conv2D(96, kernel_size=(ks, ks), activation='elu', 
			kernel_regularizer=l2(reg_term), bias_regularizer=l2(reg_term)))
		model.add(BatchNormalization())

		model.add(Conv2D(96, kernel_size=(ks, ks), activation='elu', 
			kernel_regularizer=l2(reg_term), bias_regularizer=l2(reg_term)))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.4))

		model.add(Flatten())
		model.add(BatchNormalization())
		model.add(Dense(50, activation='elu'))

		if n_classes == 2:
			model.add(Dense(1, activation='sigmoid'))
		else:
			model.add(Dense(n_classes, activation='softmax'))

		model.compile(loss=keras.losses.binary_crossentropy,
		              optimizer=keras.optimizers.Adadelta(),
		              metrics=['accuracy'])
		print(model.summary())


		# Train model
		classif_st = time.time()
		history = model.fit(X_train, y_train,
		          batch_size=32,
		          epochs=7,
		          verbose=1,
		          validation_data=(X_test, y_test))

		classif_end = time.time()
		print("Duration of classification:", round(classif_end-classif_st,2))
 
		pickle.dump(history.history, results)
		
		# Evaluate model
		score = model.evaluate(X_test, y_test, verbose=1)

		print('Test loss:', score[0])
		results.write('Test loss:'+str(score[0])+'\n')
		print('Test accuracy:', score[1])
		results.write('Test accuracy:'+str(score[1])+'\n')