from multimodal import keras as mmk
from keras.regularizers import L1L2
from keras import backend as K
from math import isclose


def test_continuity_regularizer():
    # Make an example tensor
    x = K.variable([[0, 1], [0, 1]])

    # Test axis=0
    cr = mmk.ContinuityRegularizer(1, 0)
    assert isclose(K.eval(cr(x)), 0)

    # Test axis=1
    cr = mmk.ContinuityRegularizer(1, 1)
    assert isclose(K.eval(cr(x)), 2)


def test_multi_regularizer():
    # Make an example tensor
    x = K.variable([0, 1, 2])

    # Test combining two L1L2 regularizers
    mr = mmk.MultiRegularizer([L1L2(l1=1.0), L1L2(l2=1.0)])
    assert isclose(K.eval(mr(x)), 3 + 5)
