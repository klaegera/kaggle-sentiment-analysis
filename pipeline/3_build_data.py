#!/usr/bin/env python3
import numpy
import pickle
import sys
from random import shuffle

# check if correct number of arguments supplied
if len(sys.argv) != 3:
	print('\n USAGE:', sys.argv[0], '[pos_data] [neg_data]\n')
	quit()

# load vocab
with open('vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)

print('\n *----------------------------------\n |')
print(' |  PROCESSING DATA')
print(' |\n *----------------------------------\n')

# process training data
# a data row will be:
# [ [12, 23, 34, ...], 1 ] for positive lines
# [ [12, 23, 34, ...], 0 ] for negative lines
# where 12, 23, 34, ... are the word IDs according to the vocab
data = []
with open(sys.argv[1], 'r') as f: # pos data
	for line in f:
		tokens = [vocab.get(t, -1) for t in line.strip().split()] # lookup each word in line in vocab
		tokens = [t for t in tokens if t >= 0] # keep only positive values (= words found in vocab)
		data.append([tokens, 1])
with open(sys.argv[2], 'r') as f: # neg data
	for line in f:
		tokens = [vocab.get(t, -1) for t in line.strip().split()]
		tokens = [t for t in tokens if t >= 0]
		data.append([tokens, 0])

# shuffle the data rows
shuffle(data)

# convert data to numpy ndarray
data = numpy.asarray(data)

# save data
with open('data.pkl', 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n *----------------------------------\n |')
print(' |  DATA SAVED TO: data.pkl')
print(' |\n *----------------------------------\n')

