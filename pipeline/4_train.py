#!/usr/bin/env python3
import pickle
import model
import sys

# check if correct number of arguments supplied
if len(sys.argv) != 1:
	print('\n USAGE:', sys.argv[0], '\n')
	quit()

# load weights and data
with open('weights.pkl', 'rb') as f:
	weights = pickle.load(f)
with open('data.pkl', 'rb') as f:
	data = pickle.load(f)

# instantiate Model class with loaded weights
model = model.Model(weights)

print('\n *----------------------------------\n |')
print(' |  PERFORMING CROSS-VALIDATION ON 40\'000 SAMPLES')
print(' |\n *----------------------------------\n')

# perform 4-fold cross-validation (using only the first 40'000 samples for performance)
# returns average accuracy over folds
score = model.cross_validate(data[:40000], k=4)

# round the CV score to 3 decimal places
score = round(score, 3)

print('\n *----------------------------------\n |')
print(' |  CROSS-VALIDATION RESULT:', score)
print(' | \n *----------------------------------\n |')
print(' |  TRAINING MODEL ON WHOLE DATASET')
print(' |\n *----------------------------------\n')

# train the model on all data
model = model.train(data)

# file name for saving model
# e.g. model-812.h5 for 81.2% CV accuracy
name = 'model-{:d}.h5'.format(int(score * 1000))

# save model to file
model.save(name)

print('\n *----------------------------------\n |')
print(' |  MODEL SAVED TO:', name)
print(' |\n *----------------------------------\n')

