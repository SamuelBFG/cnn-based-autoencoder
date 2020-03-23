import os

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, Flatten, Activation,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import numpy as KR
import copy
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import time
def complex_multi(h,x):

    # ---- For Complex Number multiply of h*x
    # (a+bi)*(c+di) = (ac-bd)+(bc+ad)i
    # construct h1[c,-d]
    tmp_array = KR.ones(shape=(KR.shape(x)))
    # print(KR.shape(x))
    # time.sleep(30)

    n_sign_array = KR.concatenate([tmp_array, -tmp_array], axis=1)
    h1 = h * n_sign_array

    # construct h2
    h2 = KR.reverse(h, axis=2)

    # ac - bd
    tmp = h1 * x
    h1x = KR.sum(tmp, axis=-1)

    # bc + ad
    tmp = h2 * x
    h2x = KR.sum(tmp, axis=-1)

    a_real = KR.expand_dims(h1x, axis=2)
    a_img = KR.expand_dims(h2x, axis=2)

    a_complex_array = KR.concatenate([a_real, a_img], axis=-1)

    return a_complex_array



print(complex_multi(a,b))