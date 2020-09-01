import os
import tensorflow as tf
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# print(len(text))

vocab = sorted(set(text))
# print(len(vocab))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[char] for char in text])

# print(text_as_int)

print('{')
for char, _ in zip(char2idx, range(20)):
    print('     {:4s}:  {:3d},'.format(repr(char), char2idx[char]))
print('...\n')

print('{} ----> characters mapped to int ----> {}'.format(repr(text[:13]), text_as_int[:13]))