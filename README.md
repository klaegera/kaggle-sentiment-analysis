# Kaggle Sentiment Analysis

Python pipeline for development and testing of custom Keras models used in sentiment analysis.

Created for a Kaggle competition.

## Requirements

- Bash
- Python3 + Modules:
  - NumPy
  - TensorFlow   (framework for model)
  - Keras        (abstraction layer on TensorFlow)
  - H5Py         (format to save model to disk)
  - GenSim       (Word2Vec Framework to read pre-trained word vectors)
  - SciKit-Learn (machine-learning utils, e.g. CV)

To install python3 and these modules (Debian-style) run:

```console
$ sudo apt install python3 python3-pip
$ sudo pip3 install tensorflow keras h5py gensim sklearn
```

## Pipeline

### 1. Building vocab from text files

Builds a list of words occurring in the given text files, cutting those that appear less than 5 times.

Requires: ```-```\
Creates:  ```vocab.txt```

#### Usage
```console
$ 1_build_vocab.sh [text_1] ... [text_n]
```

### 2. Pickle vocab and weights

Creates a vocab dict and initializes weights to be used in the embedding layer of the model in one of two ways:

#### Random
All words are loaded into the dict and the embedding vectors are initialized randomly with a length of 300.

#### Vectors
Uses pre-trained embeddings given as a Word2vec bin file.

All words are loaded into the dict and weights are initialized using the pre-trained embeddings or randomly, when not available.

Requires: ```vocab.txt``` (Stage 1)\
Creates:  ```vocab.pkl```, ```weights.pkl```

#### Usage
```console
$ 2_pickle_vocab_random.py
```
OR
```console
$ 2_pickle_vocab_vectors.py [word2vec_bin]
```

### 3. Build data set

Uses the vocab dict and given training data to create a data set of the form:

```python
[
  [ [ 12, 23, 34, ... ], 1 ],
  [ [ 32, 44, 21, ... ], 0 ],
  ...
]
```

whereby the first column is an array of the words in the sample, tokenized using their ID from the vocab dict, and the second column is the classification.

Classification is decided by the input files: two files are required, samples from the first are designated '1', the second '0'.
The data set is then shuffled.

Requires: ```vocab.pkl``` (Stage 2)\
Creates:  ```data.pkl```

#### Usage
```console
$ 3_build_data.py [pos_data] [neg_data]
```

### 4. Cross-validate and train model using training data

Initializes and compiles the model (described in ```model.py```) using the embedding weights created previously.

Then cross-validates (stratified 4-fold) the model on the first 40'000 samples (for duration reasons - the calculated accuracy will be conservative).

Finally, trains the model on the whole data set.

Requires: ```weights.pkl``` (Stage 2), ```data.pkl``` (Stage 3)\
Creates:  ```model-XXX.h5``` (XXX = CV score)

#### Usage
```console
$ 4_train.py
```

### 5. Predict classes of test data

Use the model trained in the previous stage to predict the classification of given test data.

The test data is first pre-processed as in Stage 3.
The result is saved in the required csv format.

Requires: ```model-XXX.h5``` (Stage 4), ```vocab.pkl``` (Stage 2)\
Creates:  ```predict.csv```

#### Usage
```console
$ 5_predict.py [model] [test_data]
```

## Example Run

Using pre-trained embeddings.

Assuming the following files:
 - ```pipeline/*```         (the scripts described above)
 - ```data/train_pos.txt``` (positive training samples)
 - ```data/train_neg.txt``` (negative training samples)
 - ```data/test_data.txt``` (test data to be predicted)
 - ```data/vectors.bin```   (pre-trained Word2Vec embeds)

### Commands
```console
$ pipeline/1_build_vocab.sh data/train_pos.txt data/train_neg.txt
$ pipeline/2_pickle_vocab_vectors.py data/vectors.bin
$ pipeline/3_build_data.py data/train_pos.txt data/train_neg.txt
$ pipeline/4_train.py
$ pipeline/5_predict.py model-XXX.h5 data/test_data.txt
```

The results will be in ```predict.csv```.
