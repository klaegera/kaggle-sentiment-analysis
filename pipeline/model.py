#!/usr/bin/env python3
from keras.models import Sequential, Model as KModel
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from keras.layers import GlobalMaxPooling1D, Concatenate, Input
from keras.preprocessing import sequence
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# pads or truncates data to constant length
def pad_data(data):
	return sequence.pad_sequences(data, maxlen=40)

# split data into arrays of samples and targets
def split_data(data):
	X = pad_data(data[:,0])
	Y = data[:,1]
	return X, Y

class Model:

	# initialize with weights used to initialize model's embedding layer
	def __init__(self, weights):
		self.weights = weights

		
	# compile and return Keras model
	def compile(self):
		model = Sequential()
		model.add(Embedding(len(self.weights), len(self.weights[0]), weights=[self.weights], trainable=True, input_length=40))
		model.add(Conv1D(512, 3, padding='same', activation='relu'))
		model.add(MaxPooling1D())
		model.add(Conv1D(512, 2, padding='same'))
		model.add(MaxPooling1D())
		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	# calculate k-fold cross-validation accuracy
	# returns mean of accuracies of each fold
	def cross_validate(self, data, k=3):
		X, Y = split_data(data)
		classifier = KerasClassifier(build_fn=self.compile, epochs=2, batch_size=128)
		result = cross_val_score(classifier, X, y=Y, cv=k)
		return sum(result) / k

	# compile, train on whole data set and return model
	def train(self, data):
		X, Y = split_data(data)
		model = self.compile()
		model.fit(X, Y, epochs=2, batch_size=128)
		return model

# run directly to develop model using grid-search
if __name__ == '__main__':
	import pickle
	from sklearn.model_selection import GridSearchCV

	with open('weights.pkl', 'rb') as f:
		weights = pickle.load(f)
	with open('data.pkl', 'rb') as f:
		data = pickle.load(f)

	model = Model(weights)
	X, Y = split_data(data[:100000])

	classifier = KerasClassifier(build_fn=model.compile, epochs=2, batch_size=128)

	grid = GridSearchCV(classifier, {'epochs':[2,3]}, refit=False, cv=3)
	grid.fit(X, Y)

	for params, mean, std in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score']):
		print(params, mean, std, 'BEST' if mean == grid.best_score_ else '')

