#!/usr/bin/env python3
import numpy
import pickle
import sys

# check if correct number of arguments supplied
if len(sys.argv) != 1:
	print('\n USAGE:', sys.argv[0], '\n')
	quit()

# assign an ID to each vocab word: {'hello': 1, 'bye': 2, ...}
vocab = {}
with open('vocab.txt', 'r') as f:
	for i, word in enumerate(f):
		vocab[word.strip()] = i + 1

# initialize embedding weights randomly
weights = numpy.random.rand(len(vocab) + 1, 300)

# save vocab and weights
with open('vocab.pkl', 'wb') as f:
	pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
with open('weights.pkl', 'wb') as f:
	pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)

print('\n *----------------------------------\n |')
print(' |  VOCAB SAVED TO: vocab.pkl')
print(' |  RANDOM WEIGHTS SAVED TO: weights.pkl')
print(' |\n *----------------------------------\n')

