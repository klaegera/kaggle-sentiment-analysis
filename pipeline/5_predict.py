#!/usr/bin/env python3
import numpy
import pickle
import sys
from keras.models import load_model
from keras.preprocessing import sequence
from model import pad_data

# check if correct number of arguments supplied
if len(sys.argv) != 3:
	print('\n USAGE:', sys.argv[0], '[model] [test_data]\n')
	quit()

# load trained model from given file
model = load_model(sys.argv[1])

# load vocab
with open('vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)

print('\n *----------------------------------\n |')
print(' |  PREPROCESSING TEST DATA')
print(' |\n *----------------------------------\n')

# process test data
# a data row will be:
# [12, 23, 34, ...]
# where 12, 23, 34, ... are the word IDs according to the vocab
data = []
with open(sys.argv[2], 'r') as f: # test data
	for line in f:
		tokens = [vocab.get(t, -1) for t in line.strip().split()] # lookup each word in line in vocab
		tokens = [t for t in tokens if t >= 0] # keep only positive values (= words found in vocab)
		data.append(tokens)

# convert test data to numpy ndarray
data = numpy.asarray(data)

# pad test data to fit model
data = pad_data(data)

print('\n *----------------------------------\n |')
print(' |  PERFORMING PREDICTION')
print(' |\n *----------------------------------\n')

# run model prediction on test data
predict = model.predict(data, verbose=1)
predict = numpy.around(predict).astype(int)

# save prediction in required format
with open('predict.csv', 'w') as f:
	f.write('Id,Prediction\n') # header line
	for i, p in enumerate(predict):
		f.write('{},{}\n'.format(i + 1, p[0] * 2 - 1)) # start at ID 1 and map result [0;1] to [-1;1]

print('\n *----------------------------------\n |')
print(' |  PREDICTION SAVED TO: predict.csv')
print(' |\n *----------------------------------\n')

