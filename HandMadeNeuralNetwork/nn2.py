import struct
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    with open('dataset/train-labels-idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))