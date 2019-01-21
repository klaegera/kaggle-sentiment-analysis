#!/usr/bin/env python3
import numpy
import gensim
import pickle
import sys

# check if correct number of arguments supplied
if len(sys.argv) != 2:
	print('\n USAGE:', sys.argv[0], '[word2vec_bin]\n')
	quit()

print('\n *----------------------------------\n |')
print(' |  LOADING WORD2VEC DATA')
print(' |\n *----------------------------------\n')

# load word vectors from given Word2Vec bin file
vecs = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)

# get word vector dimensionality
dim = vecs.vector_size

print('\n *----------------------------------\n |')
print(' |  PROCESSING DATA')
print(' |\n *----------------------------------\n')

# lookup each vocab word in Word2Vec dict
# if found, assign it an ID starting at 1 (0 used for padding) and save its vector to weights
vocab = {}
weights = []
with open('vocab.txt', 'r') as f:
	weights.append(numpy.random.rand(dim)) # word #0 is later used as padding
	for i, word in enumerate(f):
		word = word.strip() # drop newline characters
		vocab[word] = i + 1 # assign ID to word
		try:
			weights.append(vecs[word]) # raises KeyError if not found
		except KeyError:
			weights.append(numpy.random.rand(dim)) # random vector for words not in Word2Vec dict

# free memory used by Word2Vec dict
vecs = None

# convert weights to numpy ndarray
weights = numpy.asarray(weights)

# save vocab and weights
with open('vocab.pkl', 'wb') as f:
	pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
with open('weights.pkl', 'wb') as f:
	pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)

print('\n *----------------------------------\n |')
print(' |  VOCAB SAVED TO: vocab.pkl')
print(' |  VECTOR WEIGHTS SAVED TO: weights.pkl')
print(' |\n *----------------------------------\n')

