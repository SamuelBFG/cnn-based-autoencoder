from keras import backend as K
from keras.utils import to_categorical
import numpy as np
import time
import matplotlib.pyplot as plt
import copy

'''
 --- COMMUNICATION PARAMETERS ---
'''

# Bits per Symbol
k = 4

# Number of symbols
L = 50

# Channel Use
n = 1

# Effective Throughput
#  (bits per symbol)*( number of symbols) / channel use
R = k / n

# Eb/N0 used for training
train_Eb_dB = 16

# Noise Standard Deviation
noise_sigma = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))


# Number of messages used for training, each size = k*L
batch_size = 64

nb_train_word = batch_size*200


############################################
############################################
############################################


sess = K.get_session()
# Generate training binary Data
train_data = np.random.randint(low=0, high=2, size=(nb_train_word, k * L))
print('TRAIN_DATA')

print(train_data)
print(train_data.shape)
print('~# END #~'*10)

# Used as labeled data
print('TRAIN_DATA RESHAPED')
label_data = copy.copy(train_data)
train_data = np.reshape(train_data, newshape=(nb_train_word, L, k))
print(train_data)
print(train_data.shape)
print('~# END #~'*10)

# Convert Binary Data to integer
tmp_array = np.zeros(shape=k)
for i in range(k):
    tmp_array[i] = 2 ** i
int_data = tmp_array[::-1]
int_data = np.reshape(int_data, newshape=(k, 1))
print('INT_DATA')
print(int_data)
print(int_data.shape)
print('~# END #~'*10)

one_hot_data = np.dot(train_data, int_data)
vec_one_hot = to_categorical(y=one_hot_data, num_classes=2 ** k)
print('VEC_ONE_HOT')
print(vec_one_hot)
print(vec_one_hot.shape)
print('~# END #~'*10)


position = np.argmax(vec_one_hot, axis=2)
print('POSITION VECTOR')
print(position)
print(position.shape)
print('~# END #~'*10)

tmp = np.reshape(position,newshape=one_hot_data.shape)
print('TMP VECTOR')
print(tmp)
print(tmp.shape)
print('~# END #~'*10)


error_rate = np.mean(np.not_equal(one_hot_data,tmp))
print('ERROR RATE VECTOR')
print(error_rate)
print(error_rate.shape)
print('~# END #~'*10)